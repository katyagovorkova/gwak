import torch.nn as nn
import torch.nn.functional as F

class LSTM_AE(nn.Module):
    def __init__(self, num_ifos, num_timesteps, BOTTLENECK, FACTOR):
        super(LSTM_AE, self).__init__()
        print("WARNING: Change this with Eric's actual LSTM model!")
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.Conv1 = nn.Conv1d(in_channels = num_ifos, out_channels = 5, kernel_size=5, padding='same')
        self.Linear1 = nn.Linear(num_timesteps*5, 100)
        self.Linear2 = nn.Linear(100, BOTTLENECK)
        self.Linear3 = nn.Linear(BOTTLENECK, num_timesteps*5)
        self.Conv2 = nn.Conv1d(in_channels=5, out_channels = 2, kernel_size = 5, padding = 'same')
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.Conv1(x))
        x = x.view(-1, self.num_timesteps*5)
        x = F.tanh(self.Linear1(x))
        x = F.tanh(self.Linear2(x))
        x = F.tanh(self.Linear3(x))
        x = x.view(batch_size, 5, self.num_timesteps)
        x = self.Conv2(x)
        return x
