from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    n_mels: int
    n_time: int
    n_conv_layers: int
    n_mlp_layers: int
    initial_channels: int
    mlp_hidden_size: int


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = (
            nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        )

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class MelSpectrogramCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MelSpectrogramCNN, self).__init__()
        self.config = config
        self.conv_layers = nn.ModuleList()

        in_channels = 1
        out_channels = config.initial_channels
        for i in range(config.n_conv_layers):
            self.conv_layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    pool=(i < config.n_conv_layers - 1),
                )
            )
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)

        # Calculate the size of the flattened features
        self.flat_features = self._get_flat_features()

        # MLP layers
        self.mlp_layers = nn.ModuleList()
        mlp_in_features = self.flat_features
        for _ in range(config.n_mlp_layers - 1):
            self.mlp_layers.append(
                nn.Linear(mlp_in_features, config.mlp_hidden_size)
            )
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(0.1))
            mlp_in_features = config.mlp_hidden_size

        # Output layer
        self.output_layer = nn.Linear(mlp_in_features, 1)

    def _get_flat_features(self):
        x = torch.zeros(1, 1, self.config.n_mels, self.config.n_time)
        for layer in self.conv_layers:
            x = layer(x)
        return x.numel()

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.view(-1, self.flat_features)

        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)

        return self.output_layer(x)
