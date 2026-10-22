from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn


class Model(nn.Module):
    """
    Base class for different models.
    """

    def __init__(self, dim: int, dim_out=None):
        super().__init__()

        # Dims
        self.dim = dim
        self.dim_in = dim + 1
        self.dim_out = dim_out or dim
        
    def reshape_t_like_x(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        t を x のバッチ次元に合わせて (..., 1) の形に正規化する。

        許容する t の入力形状:
        - (N,)        -> (N,1)
        - (N,1)       -> そのまま
        - (...)       -> (...,1)
        - (1,)        -> (batch...,1) に broadcast
        """
        assert isinstance(t, torch.Tensor), "t must be a torch.Tensor"
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"

        # 最後に 1 次元を必ず付与
        if t.ndim == x.ndim - 1:
            t = t.unsqueeze(-1)

        # スカラー or (1,) は x に broadcast
        if t.numel() == 1:
            t = t.expand(*x.shape[:-1], 1)
        # それ以外は単純に reshape を試みる
        else:
            t = t.reshape(*x.shape[:-1], 1)

        return t

    @staticmethod
    def init_linear(
        layer: nn.Linear,
        bias_init: Callable | None = None,
        weight_init: Callable | None = None,
    ):
        if bias_init and (layer.bias is not None):
            bias_init(layer.bias)
        if weight_init:
            weight_init(layer.weight)

    # --- ここから統一ヘルパー（重要） ---
    def _check_shapes(self, t: torch.Tensor, x: torch.Tensor):
        # 期待: x[..., dim], t[..., 1] で 先頭次元が一致
        assert isinstance(t, torch.Tensor) and isinstance(x, torch.Tensor)
        assert x.shape[-1] == self.dim, f"x.shape[-1] must be {self.dim}, got {x.shape[-1]}"
        assert t.shape[-1] == 1, f"t.shape[-1] must be 1, got {t.shape[-1]}"
        assert t.shape[:-1] == x.shape[:-1], (
            f"batch dims of t {t.shape[:-1]} must match x {x.shape[:-1]}"
        )

    def _flatten_pair(self, t: torch.Tensor, x: torch.Tensor):
        # t: (..., 1), x: (..., dim) をそれぞれ (N,1), (N,dim) に変形
        self._check_shapes(t, x)
        t2 = t.reshape(-1, 1).float()
        x2 = x.reshape(-1, self.dim)
        return t2, x2

    def flatten(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        統一フラット化: t と x は同じバッチ次元で渡されることを前提にし、
        (N, 1) と (N, dim) に畳んで結合して返す
        """
        t = self.reshape_t_like_x(t, x)
        t2, x2 = self._flatten_pair(t, x)
        return torch.cat([t2, x2], dim=1)


class TimeEmbed(Model):
    def __init__(
        self,
        dim_out: int,
        activation: Callable,
        num_layers: int = 2,
        channels: int = 64,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(dim=1, dim_out=dim_out)
        self.channels = channels
        self.activation = activation
        self.register_buffer(
            "timestep_coeff",
            torch.linspace(start=0.1, end=100, steps=self.channels).unsqueeze(0),
            persistent=False,
        )
        self.timestep_phase = nn.Parameter(torch.randn(1, self.channels))
        self.hidden_layer = nn.ModuleList([nn.Linear(2 * self.channels, self.channels)])
        self.hidden_layer += [
            nn.Linear(self.channels, self.channels) for _ in range(num_layers - 2)
        ]
        self.out_layer = nn.Linear(self.channels, self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )

    def forward(self, t: torch.Tensor, *args) -> torch.Tensor:
        # 期待: t は (..., 1)
        assert t.ndim >= 2 and t.shape[-1] == 1, "t must have shape (..., 1)"
        t2 = t.reshape(-1, 1).float()  # (N,1)
        sin_embed_t = torch.sin((self.timestep_coeff * t2) + self.timestep_phase)
        cos_embed_t = torch.cos((self.timestep_coeff * t2) + self.timestep_phase)
        assert cos_embed_t.shape == (t2.shape[0], self.channels)
        embed_t = torch.cat([sin_embed_t, cos_embed_t], dim=1)
        for layer in self.hidden_layer:
            embed_t = self.activation(layer(embed_t))
        return self.out_layer(embed_t)  # (N, dim_out)

class _StraightThroughClip(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(tensor, max_norm):
        return tensor.clamp(min=-max_norm, max=max_norm)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None

    @staticmethod
    def jvp(ctx, gradient, limit_gradient):
        return gradient


def clip(tensor: torch.Tensor, max_norm: float | None = None, gradient_mode="straight_through"):
    """Straight-through per-component clip -- see egnn.py's clip() for the
    identical fix and full rationale (a plain torch.clip zeros the gradient
    beyond its bounds, silently killing the training signal on the network's
    own output exactly when a large-magnitude control is needed)."""
    if max_norm is not None:
        if gradient_mode == "exact":
            return tensor.clip(min=-1.0 * max_norm, max=max_norm)
        # Subtracting and readding large values loses the clipped result in
        # float32. Return the actual clamp while preserving its ST derivative.
        tensor = _StraightThroughClip.apply(tensor, max_norm)
    return tensor


class FlattenReshapeWrapper(nn.Module):
    """
    Wraps a flat-vector model (e.g. FourierMLP) so it can be used on particle-shaped
    data of shape (..., n_particles, spatial_dim): flattens to (..., n_particles *
    spatial_dim) before calling the wrapped model, and reshapes its output back to the
    particle-shaped layout afterward. This lets a non-equivariant flat MLP see the full
    joint configuration (all particles at once) rather than being applied per-particle.
    """

    def __init__(self, base_model: nn.Module, n_particles: int, spatial_dim: int):
        super().__init__()
        self.base_model = base_model
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        prefix = x.shape[:-2]
        x_flat = x.reshape(*prefix, self.n_particles * self.spatial_dim)
        out_flat = self.base_model(t, x_flat)
        return out_flat.reshape(*prefix, self.n_particles, self.spatial_dim)


class FourierMLP(Model):
    def __init__(
        self,
        dim: int,
        activation: Callable = torch.nn.GELU,
        num_layers: int = 4,
        channels: int = 64,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
        clip_norm: float | None = None,
        time_coordinates: str = "physical",
        clip_gradient_mode: str = "straight_through",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.channels = channels
        self.activation = activation()
        self.input_embed = nn.Linear(self.dim, self.channels)
        self.timestep_embed = TimeEmbed(
            dim_out=self.channels,
            activation=self.activation,
            num_layers=2,
            channels=self.channels,
        )
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(self.channels, self.channels) for _ in range(num_layers - 2)]
        )
        self.out_layer = nn.Linear(self.channels, self.dim_out)
        Model.init_linear(
            self.out_layer,
            bias_init=torch.nn.init.zeros_,
            weight_init=torch.nn.init.zeros_,
        )
        
        self.clip_norm = clip_norm
        if clip_gradient_mode not in ("straight_through", "exact"):
            raise ValueError("Invalid clip_gradient_mode")
        self.clip_gradient_mode = clip_gradient_mode
        if time_coordinates not in ("physical", "variance", "both"):
            raise ValueError("time_coordinates must be physical, variance, or both")
        self.time_coordinates = time_coordinates
        self._time_reference = None
        if time_coordinates == "both":
            self.variance_timestep_embed = TimeEmbed(
                dim_out=self.channels, activation=self.activation,
                num_layers=2, channels=self.channels)

    def set_reference(self, reference):
        # Preserve legacy state_dict shapes in physical-time mode.
        if self.time_coordinates != "physical":
            self._time_reference = reference

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 期待: x は (..., dim), t は同じ先頭次元で (..., 1)
        t = self.reshape_t_like_x(t, x)
        self._check_shapes(t, x)

        x_dim = x.dim()
        prefix = x.shape[:-1]  # 例: (B,) or (B, L)

        # 2D に畳んで処理
        t2, x2 = self._flatten_pair(t, x)          # (N,1), (N,dim)
        if self.time_coordinates == "physical":
            embed_t = self.timestep_embed(t2)
        else:
            if self._time_reference is None:
                raise RuntimeError("Variance-time embedding requires set_reference()")
            tau = self._time_reference.reparametrize_time(t2).to(t2)
            if self.time_coordinates == "variance":
                embed_t = self.timestep_embed(tau)
            else:
                embed_t = self.timestep_embed(t2)+self.variance_timestep_embed(tau)
        embed_x = self.input_embed(x2)             # (N, C)
        assert embed_t.shape == embed_x.shape

        embed = embed_x + embed_t
        for layer in self.hidden_layer:
            embed = layer(self.activation(embed))

        out = self.out_layer(self.activation(embed))  # (N, dim_out)

        # 入力が 3D の場合のみ元形状に戻す（従来挙動を維持）
        if x_dim == 3:
            out = out.view(*prefix, self.dim_out)
            
        if self.clip_norm is not None:
            out = clip(out, max_norm=self.clip_norm, gradient_mode=self.clip_gradient_mode)
        return out


class FeedForward(Model):
    def __init__(
        self,
        dim: int,
        arch: Sequence[int],
        activation: Callable,
        normalization_factory=None,
        normalization_kwargs=None,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)

        # Affine linear layer
        bias = normalization_factory is None
        self.hidden_layer = nn.ModuleList([nn.Linear(self.dim_in, arch[0], bias=bias)])
        self.hidden_layer += [
            nn.Linear(arch[i], arch[i + 1], bias=bias) for i in range(len(arch) - 1)
        ]
        self.out_layer = nn.Linear(arch[-1], self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )

        # Activation function
        self.activation = activation

        # Normalization layer
        if normalization_factory:
            normalization_kwargs = normalization_kwargs or {}
            self.norm_layer = nn.ModuleList(
                [
                    normalization_factory(num_features, **normalization_kwargs)
                    for num_features in arch
                ]
            )
        else:
            self.norm_layer = None

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tensor = self.flatten(t, x)  # (N, 1+dim)

        for i, linear in enumerate(self.hidden_layer):
            tensor = self.activation(linear(tensor))
            if self.norm_layer is not None:
                tensor = self.norm_layer[i](tensor)

        return self.out_layer(tensor)  # (N, dim_out)


class DenseNet(Model):
    def __init__(
        self,
        dim: int,
        arch: list[int],
        activation: Callable,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.nn_dims = [self.dim_in] + arch
        self.hidden_layer = nn.ModuleList(
            [
                nn.Linear(sum(self.nn_dims[: i + 1]), self.nn_dims[i + 1])
                for i in range(len(self.nn_dims) - 1)
            ]
        )
        self.out_layer = nn.Linear(sum(self.nn_dims), self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )
        self.activation = activation

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tensor = self.flatten(t, x)  # (N, 1+dim)
        for layer in self.hidden_layer:
            tensor = torch.cat([tensor, self.activation(layer(tensor))], dim=1)
        return self.out_layer(tensor)  # (N, dim_out)


class LevelNet(Model):
    """
    Network module for a single level
    """

    def __init__(
        self,
        dim,
        dim_embed,
        level,
        activation,
        normalization_factory=None,
        normalization_kwargs=None,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)

        self.level = level
        bias = normalization_factory is None
        self.dense_layers = nn.ModuleList(
            [nn.Linear(self.dim_in, dim_embed, bias=bias)]
        )
        self.dense_layers += [
            nn.Linear(dim_embed, dim_embed, bias=bias) for _ in range(2**level - 1)
        ]
        self.dense_layers.append(nn.Linear(dim_embed, self.dim_out))
        Model.init_linear(
            self.dense_layers[-1],
            bias_init=last_bias_init,
            weight_init=last_weight_init,
        )  # type: ignore
        if normalization_factory is None:
            self.norm_layers = None
        else:
            normalization_kwargs = normalization_kwargs or {}
            self.norm_layers = nn.ModuleList(
                [
                    normalization_factory(dim_embed, **normalization_kwargs)
                    for _ in range(2**level)
                ]
            )
        self.act = activation

    def forward(self, t, x, res_tensors=None):
        tensor = self.flatten(t, x)  # (N, 1+dim)
        out_tensors = []
        tensor = self.dense_layers[0](tensor)
        for i, dense in enumerate(self.dense_layers[1:]):  # type: ignore
            if self.norm_layers is not None:
                tensor = self.norm_layers[i](tensor)
            tensor = self.act(tensor)
            tensor = dense(tensor)
            if res_tensors:
                tensor = tensor + res_tensors[i]
            if i % 2 or self.level == 0:
                out_tensors.append(tensor)
        return out_tensors


class MultilevelNet(Model):
    """
    Multilevel net
    """

    def __init__(
        self,
        dim,
        activation,
        factor=5,
        levels=4,
        normalization_factory=None,
        normalization_kwargs=None,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.nets = nn.ModuleList(
            [
                LevelNet(
                    dim=dim,
                    dim_embed=factor * self.dim_in,
                    level=level,
                    activation=activation,
                    normalization_factory=normalization_factory,
                    normalization_kwargs=normalization_kwargs,
                    last_bias_init=last_bias_init,
                    last_weight_init=last_weight_init,
                    **kwargs,
                )
                for level in range(levels)
            ]
        )

    def forward(self, t, x):
        # ここでも t と x の形は一致している前提
        res_tensors = None
        for net in self.nets[::-1]:  # type: ignore
            res_tensors = net(t, x, res_tensors)
        assert res_tensors is not None
        return res_tensors[-1]
