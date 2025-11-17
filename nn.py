import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, num_actions)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x