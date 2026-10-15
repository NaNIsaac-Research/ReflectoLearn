import numpy as np
import torch
import torch.nn as nn


class FourierFeatureMapping(nn.Module):
    """Random Fourier Features.

    Maps the coordinate (q) into a high-dimensional frequency space to help the
    model learn high-frequency patterns.
    """
    def __init__(self, input_dim=1, mapping_size=64, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_proj = x.transpose(1, 2) @ self.B
        x_proj = x_proj.transpose(1, 2) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

class XRRPhysicsModel(nn.Module):
    def __init__(self, q_len: int, input_channels: int = 2, output_dim: int = 3,
                n_channels: int = 64, depth: int = 5, mlp_hidden: int = 256,
                dropout: float = 0.1, use_fourier: bool = True, fourier_scale: float = 10.0,
                fourier_features: int = 32):
        super().__init__()
        self.use_fourier = use_fourier
        inplace: bool = True  # False while debugging
        # Store config for checkpoint compatibility.
        self.config = {
            'q_len': q_len, 'input_channels': input_channels, 'output_dim': output_dim,
            'n_channels': n_channels, 'depth': depth, 'mlp_hidden': mlp_hidden,
            'dropout': dropout, 'use_fourier': use_fourier, 'fourier_scale': fourier_scale,
            'fourier_features': fourier_features
        }

        self.fourier = FourierFeatureMapping(input_dim=1, mapping_size=fourier_features, scale=fourier_scale)
        enc_in_channels = input_channels + (self.fourier.mapping_size * 2) if use_fourier else input_channels
        self.register_buffer('q_grid', torch.linspace(0, 1, q_len).view(1, 1, -1))

        layers = []
        curr_dim = enc_in_channels
        for i in range(depth):
            out_dim = n_channels * (2 ** min(i, 3))
            layers.append(nn.Sequential(
                nn.Conv1d(curr_dim, out_dim, kernel_size=7, padding=3, bias=False),
                # Lower momentum (0.1 -> 0.03): with streaming augmentation the
                # batch variance can be large, so the running stats converge
                # slowly and stably, preventing eval-time blow-up.
                nn.BatchNorm1d(out_dim, momentum=0.03),
                nn.LeakyReLU(0.2, inplace=inplace),
                nn.Dropout1d(dropout),
                nn.MaxPool1d(kernel_size=2)
            ))
            curr_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, mlp_hidden),
            nn.LeakyReLU(0.2, inplace=inplace),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LeakyReLU(0.2, inplace=inplace),
            nn.Linear(mlp_hidden // 2, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        bs = x.shape[0]
        if self.use_fourier:
            q_batch = self.q_grid.expand(bs, -1, -1)
            q_fourier = self.fourier(q_batch)
            x = torch.cat([x, q_fourier], dim=1)
        feat = self.encoder(x)
        feat = self.global_pool(feat).squeeze(-1)
        return self.regressor(feat)
