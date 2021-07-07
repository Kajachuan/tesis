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
        self.blstm = nn.LSTM(input_size=channels, hidden_size=4, num_layers=2,
                             batch_first=True, dropout=0.3)
        self.linear = nn.Linear(8, channels)
        self.activation = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input Dim = (batch, channels, timesteps)
        data = input.transpose(1, 2) # Dim = (batch, timesteps, channels)
        self.blstm.flatten_parameters()
        data = self.blstm(data)[0] # Dim = (batch, timesteps, 8)
        data = self.linear(data) # Dim = (batch, timesteps, channels)
        data = data.transpose(1, 2) # Dim = (batch, channels, timesteps)
        mask = self.activation(data) # Dim = (batch, channels, timesteps)

        estimates = input * mask
        return estimates
