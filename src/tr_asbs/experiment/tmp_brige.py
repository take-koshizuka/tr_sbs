import torch
from tqdm import tqdm

from tr_soc.common.lagrangian_utils import optimize_lm
from tr_soc.common.noise_scheduler import create_cos_square_scheduler
from tr_soc.common.prior import Prior
from tr_soc.networks.mlp import FourierMLP


class TrustRegionLvBuffer:
    def __init__(self, cfg, target, device):
        self.buffer = None
        self.cfg = cfg
        self.device = device
        self.num_steps = cfg.num_steps
        self.batch_size = cfg.batch_size

        self.network = FourierMLP(dim=target.dim).to(device)
        self.step_size = cfg.step_size
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.step_size)
        self.grad_clip = cfg.grad_clip

        self.outer_steps = cfg.outer_steps
        self.inner_steps = cfg.inner_steps

        self.dim = target.dim

        T = cfg.T
        dt = T / cfg.num_steps

        beta_max = 8
        beta_min = 0.01

        self.eta = cfg.init_std
        schedules = create_cos_square_scheduler(C_start=beta_max, C_end=beta_min, T=cfg.T)
        scheduler, forward_integrated_scheduler, delta_integrated_scheduler, self.backward_integrated_scheduler = schedules
        self.step_tensor = torch.Tensor([step for step in range(self.num_steps + 1)]).unsqueeze(1).to(device)

        self.noise_bwd_int = torch.exp(
            -torch.Tensor([self.backward_integrated_scheduler(dt * step) for step in range(self.num_steps)]).to(device))

        self.alpha_vals = 1 - torch.exp(-2 * torch.Tensor(
            [delta_integrated_scheduler(step * dt, (step + 1) * dt) for step in range(cfg.num_steps + 1)])).to(
            self.device)

        self.kappa_vals = torch.Tensor(
            [(self.eta * (1 - torch.sqrt(1 - self.alpha_vals[step]))) ** 2 / self.alpha_vals[step] for step in
             range(cfg.num_steps + 1)]).to(self.device)

        # mean = torch.zeros(self.dim).to(self.device)
        # std = torch.ones(self.dim).to(self.device) * self.eta
        self.prior = Prior(self.dim, self.eta, self.device)
        # self.prior_score = lambda x: -x / self.eta ** 2

        self.target = target

        self.lm_prod = 1.
        self.lm = 1.

    def fill_buffer(self):
        self.buffer = ReplayBuffer(self.cfg, self.device)
        for _ in tqdm(range(self.cfg.buffer_max_length // self.batch_size + 1), desc='Collecting data'):
            x, ctrls, noises, log_weights, nabla_g = self.sde_integrate()
            self.buffer.add((x, ctrls, noises, log_weights, nabla_g))

    def sde_integrate(self):
        x_t = torch.zeros(
            (self.batch_size, self.num_steps + 1, self.dim),
            device=self.device,
        )
        log_weights = torch.zeros(
            (self.batch_size,),
            device=self.device,
        )

        noises = torch.normal(
            torch.zeros(
                (self.batch_size, self.num_steps, self.dim),
                device=self.device,
            ),
            torch.ones(
                (self.batch_size, self.num_steps, self.dim),
                device=self.device,
            ),
        )
        ctrls = torch.normal(
            torch.zeros(
                (self.batch_size, self.num_steps, self.dim),
                device=self.device,
            ),
        )

        x = self.prior.sample(self.batch_size).to(self.device)
        x_t[:, 0] = x

        for s in range(self.num_steps):
            alpha = self.alpha_vals[s]
            kappa = self.kappa_vals[s]
            step = torch.ones([x.shape[0], 1]).to(self.device) * s
            ctrl = self.network(step, x)

            means = x + (1 - torch.sqrt(1 - alpha)) * (2 * self.eta ** 2 * ctrl - x)
            x = means + self.eta * torch.sqrt(alpha) * noises[:, s]
            log_weights_deterministic = -2 * kappa * torch.sum(ctrl ** 2, -1)
            log_weights_stochastic = -2 * torch.sqrt(kappa) * torch.sum(ctrl * noises[:, s], -1)
            log_weights += log_weights_deterministic + log_weights_stochastic

            x_t[:, s + 1] = x
            ctrls[:, s] = ctrl

        log_weights += self.target.log_prob(x) - self.prior.log_prob(x)
        nabla_g = self.prior.score(x) - self.target.score(x)

        return x_t.detach(), ctrls.detach(), noises.detach(), log_weights.detach(), nabla_g.detach()

    def train(self):
        self.lm = optimize_lm(self.buffer.get_log_weights(), eps_tr=self.cfg.trust_region_bound)
        self.lm_prod = self.lm_prod * (self.lm / (1. + self.lm))
        weight_scale = 1. / (1. + self.lm)
        annealing = 1 - self.lm_prod

        loss_sum = 0
        for _ in tqdm(range(self.inner_steps), desc="Training"):

            x, detached_ctrls, noises, log_weights, nabla_g = self.buffer.sample()

            log_importance_weight = weight_scale * log_weights
            ctrls = self.network(self.step_tensor, x).view(x.shape)
            det_integrand = 2 * torch.sum(self.kappa_vals[:-1][:,None] * (detached_ctrls - ctrls[:,:-1]) ** 2, dim=(1, 2))
            stochastic_integrand = 2 * torch.sum(torch.sqrt(self.kappa_vals[:-1][:,None]) * noises * (detached_ctrls - ctrls[:,:-1]), dim=(1, 2))

            log_rnds = log_importance_weight + det_integrand + stochastic_integrand
            loss = torch.var(log_rnds)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_sum += loss.item()

        loss = loss_sum / self.inner_steps

        return loss