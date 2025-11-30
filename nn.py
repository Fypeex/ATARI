import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Simple DQN convolutional network for 84x84 grayscale Atari frames.

    Expects input of shape (B, in_channels=4, 84, 84) with float values in [0,1].
    Outputs Q-values for each action.
    """

    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x