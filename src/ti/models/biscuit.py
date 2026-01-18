from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ti.models.encoders import ActionEmbed, BiscuitDecoder, BiscuitEncoder


class BISCUIT_VAE(nn.Module):
    def __init__(
        self,
        obs_dim,
        z_dim,
        n_actions,
        hidden_dim,
        a_dim=8,
        tau_start=1.0,
        tau_end=5.0,
        interaction_reg_weight=5e-4,
        beta_kl=1.0,
        device=None,
        lr=3e-4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.beta_kl = float(beta_kl)
        self.device = device

        self.enc = BiscuitEncoder(obs_dim, z_dim, hidden_dim).to(device)
        self.dec = BiscuitDecoder(obs_dim, z_dim, hidden_dim).to(device)
        self.aemb = ActionEmbed(n_actions, a_dim).to(device)

        self.mlp_I = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(z_dim + a_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(z_dim)
            ]
        ).to(device)

        self.mlp_prior = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(z_dim + 1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2),
                )
                for _ in range(z_dim)
            ]
        ).to(device)

        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.interaction_reg_weight = float(interaction_reg_weight)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    @staticmethod
    def _reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _tau(self, step, total_steps):
        if total_steps <= 1:
            return self.tau_end
        frac = float(step) / float(total_steps - 1)
        return self.tau_start + frac * (self.tau_end - self.tau_start)

    def _predict_Ihat(self, z_prev, a_emb, tau):
        inp = torch.cat([z_prev, a_emb], dim=-1)
        logits = torch.cat([mlp(inp) for mlp in self.mlp_I], dim=-1)
        I_hat = torch.tanh(logits * tau)
        return I_hat

    def _prior_params(self, z_prev, I_hat):
        mus, logvars = [], []
        for i in range(self.z_dim):
            out = self.mlp_prior[i](torch.cat([z_prev, I_hat[:, i : i + 1]], dim=-1))
            mus.append(out[:, 0:1])
            logvars.append(out[:, 1:2])
        mu_p = torch.cat(mus, dim=-1)
        logvar_p = torch.cat(logvars, dim=-1).clamp(-10.0, 10.0)
        return mu_p, logvar_p

    @torch.no_grad()
    def encode_mean(self, x):
        self.eval()
        mu, _ = self.enc(x)
        return mu

    def train_step(self, x_prev, a, x_t, step_idx, total_steps, use_amp=False, amp_dtype="bf16", scaler=None):
        self.train()
        tau = self._tau(step_idx, total_steps)
        use_amp = bool(use_amp) and self.device is not None and self.device.type == "cuda"
        amp_dtype = str(amp_dtype).lower()
        if amp_dtype in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif amp_dtype in ("fp16", "float16"):
            dtype = torch.float16
        else:
            raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
            if use_amp
            else nullcontext()
        )

        with autocast_ctx:
            mu_prev, logvar_prev = self.enc(x_prev)
            z_prev = self._reparam(mu_prev, logvar_prev)

            mu_q, logvar_q = self.enc(x_t)
            z_t = self._reparam(mu_q, logvar_q)

            x_hat = self.dec(z_t)
            recon = F.mse_loss(x_hat, x_t, reduction="none").sum(dim=-1).mean()

            a_emb = self.aemb(a)
            I_hat = self._predict_Ihat(z_prev, a_emb, tau)
            mu_p, logvar_p = self._prior_params(z_prev, I_hat)

            var_q = torch.exp(logvar_q)
            var_p = torch.exp(logvar_p)
            kl = 0.5 * (
                logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
            ).sum(dim=-1).mean()

            inter_reg = torch.clamp(I_hat + 1.0, min=0.0).pow(2).mean()
            loss = recon + self.beta_kl * kl + self.interaction_reg_weight * inter_reg

        self.opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(self.opt)
            scaler.update()
        else:
            loss.backward()
            self.opt.step()
        return loss
