import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ti.data.buffer import build_episode_index_strided
from ti.data.collect import collect_offline_dataset
from ti.envs import PeriodicMaze
from ti.envs import layouts
from ti.models.rep_methods import OfflineRepLearner
from ti.utils import seed_everything

from periodicity_study.common import ensure_dir, maze_cfg_from_config


def _decode_phase_index(obs: torch.Tensor, period: int) -> torch.Tensor:
    nuis = obs[:, 2:5]
    phases = torch.arange(period, device=obs.device)
    codebook = layouts.phase_sincos3(phases, period)
    dists = torch.cdist(nuis, codebook)
    return dists.argmin(dim=1)


@dataclass
class BaseRepresentation:
    name: str
    dim: int

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CoordOnlyRep(BaseRepresentation):
    def __init__(self):
        super().__init__(name="coord_only", dim=2)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, :2]


def _decode_slippery_index(obs: torch.Tensor) -> torch.Tensor:
    return obs[:, 2:5].argmax(dim=1)


def _decode_teacup_phase(obs: torch.Tensor, device: torch.device) -> torch.Tensor:
    feat = obs[:, 2:5]
    codebook = layouts.TETRA4.to(device)
    dists = torch.cdist(feat, codebook)
    idx = dists.argmin(dim=1)
    zero_mask = feat.abs().sum(dim=1) < 1e-6
    idx = torch.where(zero_mask, torch.zeros_like(idx), idx)
    return idx


class CoordNuisanceRep(BaseRepresentation):
    def __init__(self, env_id: str, period: int, device: torch.device):
        super().__init__(name="coord_plus_nuisance", dim=3)
        self.env_id = env_id
        self.period = int(period)
        self.device = device

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if self.env_id.startswith("slippery"):
            nuis_idx = _decode_slippery_index(obs).float().unsqueeze(1)
        elif self.env_id.startswith("teacup"):
            nuis_idx = _decode_teacup_phase(obs, self.device).float().unsqueeze(1)
        else:
            nuis_idx = _decode_phase_index(obs, self.period).float().unsqueeze(1)
        return torch.cat([obs[:, :2], nuis_idx], dim=1)


class CRTRRep(BaseRepresentation):
    def __init__(self, encoder: torch.nn.Module, dim: int):
        super().__init__(name="crtr_learned", dim=int(dim))
        self.encoder = encoder
        self.encoder.eval()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(obs)


class IDMRep(BaseRepresentation):
    def __init__(self, encoder: torch.nn.Module, dim: int):
        super().__init__(name="idm_learned", dim=int(dim))
        self.encoder = encoder
        self.encoder.eval()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(obs)


class OnlineCRTRRep(BaseRepresentation):
    def __init__(self, learner: OfflineRepLearner, dim: int, device: torch.device):
        super().__init__(name="crtr_online_joint", dim=int(dim))
        self.learner = learner
        self.device = device

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.learner.rep_enc(obs)

    def update(self, buf, batch_size: int, steps: int = 1) -> float:
        if buf.size < batch_size:
            return float("nan")
        epi = build_episode_index_strided(buf.timestep, buf.size, buf.num_envs, self.device)
        if epi.num_episodes == 0:
            return float("nan")
        self.learner.train()
        last_loss = float("nan")
        for _ in range(int(steps)):
            loss = self.learner.loss(buf, epi, batch_size)
            self.learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.learner.opt.step()
            last_loss = float(loss.item())
        return last_loss


class OnlineIDMRep(BaseRepresentation):
    def __init__(self, learner: OfflineRepLearner, dim: int, device: torch.device):
        super().__init__(name="idm_online_joint", dim=int(dim))
        self.learner = learner
        self.device = device

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.learner.rep_enc(obs)

    def update(self, buf, batch_size: int, steps: int = 1) -> float:
        if buf.size < batch_size:
            return float("nan")
        self.learner.train()
        last_loss = float("nan")
        for _ in range(int(steps)):
            loss = self.learner.loss(buf, None, batch_size)
            self.learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.learner.opt.step()
            last_loss = float(loss.item())
        return last_loss


def train_or_load_crtr(
    cfg,
    device: torch.device,
    cache_dir: str,
    force_retrain: bool = False,
    buf=None,
    epi=None,
) -> CRTRRep:
    ensure_dir(cache_dir)
    model_path = os.path.join(cache_dir, "crtr_rep.pt")

    if os.path.exists(model_path) and not force_retrain:
        payload = torch.load(model_path, map_location=device)
        encoder = OfflineRepLearner(
            "CRTR",
            obs_dim=cfg.obs_dim,
            z_dim=cfg.z_dim,
            hidden_dim=cfg.hidden_dim,
            n_actions=cfg.n_actions,
            crtr_temp=cfg.crtr_temp,
            crtr_rep=cfg.crtr_rep_factor,
            k_cap=cfg.k_cap,
            geom_p=cfg.geom_p,
            device=device,
            lr=cfg.lr,
        ).rep_enc.to(device)
        encoder.load_state_dict(payload["encoder"])
        return CRTRRep(encoder, dim=cfg.z_dim)

    seed_everything(int(cfg.seed), deterministic=True)
    maze_cfg = maze_cfg_from_config(cfg)
    if buf is None:
        buf, _ = collect_offline_dataset(
            PeriodicMaze,
            cfg.offline_collect_steps,
            cfg.offline_num_envs,
            maze_cfg,
            device,
        )
    if epi is None:
        epi = build_episode_index_strided(buf.timestep, buf.size, cfg.offline_num_envs, device)

    learner = OfflineRepLearner(
        "CRTR",
        obs_dim=cfg.obs_dim,
        z_dim=cfg.z_dim,
        hidden_dim=cfg.hidden_dim,
        n_actions=cfg.n_actions,
        crtr_temp=cfg.crtr_temp,
        crtr_rep=cfg.crtr_rep_factor,
        k_cap=cfg.k_cap,
        geom_p=cfg.geom_p,
        device=device,
        lr=cfg.lr,
    ).to(device)

    learner.train_steps(
        buf,
        epi,
        cfg.offline_train_steps,
        cfg.offline_batch_size,
        cfg.print_train_every,
        use_amp=False,
        amp_dtype="bf16",
        resume=False,
    )

    torch.save({"encoder": learner.rep_enc.state_dict()}, model_path)
    return CRTRRep(learner.rep_enc, dim=cfg.z_dim)


def train_or_load_idm(
    cfg,
    device: torch.device,
    cache_dir: str,
    force_retrain: bool = False,
    buf=None,
    epi=None,
) -> IDMRep:
    ensure_dir(cache_dir)
    model_path = os.path.join(cache_dir, "idm_rep.pt")

    if os.path.exists(model_path) and not force_retrain:
        payload = torch.load(model_path, map_location=device)
        encoder = OfflineRepLearner(
            "IDM",
            obs_dim=cfg.obs_dim,
            z_dim=cfg.z_dim,
            hidden_dim=cfg.hidden_dim,
            n_actions=cfg.n_actions,
            crtr_temp=cfg.crtr_temp,
            crtr_rep=cfg.crtr_rep_factor,
            k_cap=cfg.k_cap,
            geom_p=cfg.geom_p,
            device=device,
            lr=cfg.lr,
        ).rep_enc.to(device)
        encoder.load_state_dict(payload["encoder"])
        return IDMRep(encoder, dim=cfg.z_dim)

    seed_everything(int(cfg.seed), deterministic=True)
    maze_cfg = maze_cfg_from_config(cfg)
    if buf is None:
        buf, _ = collect_offline_dataset(
            PeriodicMaze,
            cfg.offline_collect_steps,
            cfg.offline_num_envs,
            maze_cfg,
            device,
        )
    if epi is None:
        epi = build_episode_index_strided(buf.timestep, buf.size, cfg.offline_num_envs, device)

    learner = OfflineRepLearner(
        "IDM",
        obs_dim=cfg.obs_dim,
        z_dim=cfg.z_dim,
        hidden_dim=cfg.hidden_dim,
        n_actions=cfg.n_actions,
        crtr_temp=cfg.crtr_temp,
        crtr_rep=cfg.crtr_rep_factor,
        k_cap=cfg.k_cap,
        geom_p=cfg.geom_p,
        device=device,
        lr=cfg.lr,
    ).to(device)

    learner.train_steps(
        buf,
        epi,
        cfg.offline_train_steps,
        cfg.offline_batch_size,
        cfg.print_train_every,
        use_amp=False,
        amp_dtype="bf16",
        resume=False,
    )

    torch.save({"encoder": learner.rep_enc.state_dict()}, model_path)
    return IDMRep(learner.rep_enc, dim=cfg.z_dim)


def init_online_crtr(cfg, device: torch.device) -> OnlineCRTRRep:
    learner = OfflineRepLearner(
        "CRTR",
        obs_dim=cfg.obs_dim,
        z_dim=cfg.z_dim,
        hidden_dim=cfg.hidden_dim,
        n_actions=cfg.n_actions,
        crtr_temp=cfg.crtr_temp,
        crtr_rep=cfg.crtr_rep_factor,
        k_cap=cfg.k_cap,
        geom_p=cfg.geom_p,
        device=device,
        lr=cfg.lr,
    ).to(device)
    return OnlineCRTRRep(learner, dim=cfg.z_dim, device=device)


def init_online_idm(cfg, device: torch.device) -> OnlineIDMRep:
    learner = OfflineRepLearner(
        "IDM",
        obs_dim=cfg.obs_dim,
        z_dim=cfg.z_dim,
        hidden_dim=cfg.hidden_dim,
        n_actions=cfg.n_actions,
        crtr_temp=cfg.crtr_temp,
        crtr_rep=cfg.crtr_rep_factor,
        k_cap=cfg.k_cap,
        geom_p=cfg.geom_p,
        device=device,
        lr=cfg.lr,
    ).to(device)
    return OnlineIDMRep(learner, dim=cfg.z_dim, device=device)
