import torch
import torch.nn as nn
import torch._dynamo.config
import torch._inductor.config

from typing import List
from dataclasses import dataclass

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True


@dataclass
class ModelConfig:
    n_mels: int
    n_time: int
    n_conv_layers: int
    n_mlp_layers: int
    initial_channels: int
    mlp_hidden_size: int


class MelSpectrogramCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MelSpectrogramCNN, self).__init__()
        self.config = config

        conv_layers: List[nn.Module] = []
        in_channels = 1
        out_channels = config.initial_channels
        for i in range(config.n_conv_layers):
            conv_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    (
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        if i < config.n_conv_layers - 1
                        else nn.Identity()
                    ),
                ]
            )
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flat_features = self._get_flat_features()

        mlp_layers: List[nn.Module] = []
        mlp_in_features = self.flat_features
        for _ in range(config.n_mlp_layers - 1):
            mlp_layers.extend(
                [
                    nn.Linear(mlp_in_features, config.mlp_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            mlp_in_features = config.mlp_hidden_size
        self.mlp_layers = nn.Sequential(*mlp_layers)

        self.output_layer = nn.Linear(mlp_in_features, 1)

    def _get_flat_features(self):
        x = torch.zeros(1, 1, self.config.n_mels, self.config.n_time)
        x = self.conv_layers(x)
        return x.numel()

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.mlp_layers(x)
        return self.output_layer(x)
