import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs):
        output = F.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(inputs)
        return F.relu(output)


class GridEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, feature_dim, 2, stride=2)
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        output = F.relu(self.bn1(self.conv1(inputs)))
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_tokens=64):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.adaptive_pool = nn.AdaptiveAvgPool2d((int(num_tokens ** 0.5), int(num_tokens ** 0.5)))
        self.projection = nn.Linear(input_dim, output_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs):
        batch_size, channels, _, _ = inputs.shape
        output = self.adaptive_pool(inputs)
        output = output.view(batch_size, channels, -1).transpose(1, 2)
        output = self.projection(output)
        output = output + self.pos_embedding
        return self.norm(output)
