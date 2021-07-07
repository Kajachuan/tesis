import torch
import torch.nn as nn
from utils.stft import STFT

class BlendNet(nn.Module):
    def __init__(self, channels: int, nfft: int, hop: int) -> None:
        super(BlendNet, self).__init__()

        self.bins = nfft // 2 + 1
        self.nfft = nfft
        self.hop = hop
        self.channels = channels
        self.stft = STFT(nfft, hop)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=16, kernel_size=3, padding=1)
        self.blstm = nn.LSTM(input_size=self.bins * channels, hidden_size=4, num_layers=2,
                             batch_first=True, dropout=0.3, bidirectional=True)
        self.linear = nn.Linear(16, channels)
        self.activation = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input Dim = (batch, channels, timesteps)
        data = self.conv(input) # Dim = (batch, 16, timesteps)
        data = data.transpose(1, 2) # Dim = (batch, timesteps, 16)
        data = self.linear(data) # Dim = (batch, timesteps, channels)
        data = data.transpose(1, 2) # Dim = (batch, channels, timesteps)
        mask = self.activation(data) # Dim = (batch, channels, timesteps)

        estimates = input * mask
        return estimates
