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
                             batch_first=False, dropout=0.3, bidirectional=True)
        self.linear = nn.Linear(8, channels)
        self.activation = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input Dim = (batch, channels, timesteps)
        data = data.permute(2, 0, 1)
        self.blstm.flatten_parameters()
        data = self.blstm(data)[0] # Dim = (timesteps, batch, 8)
        data = data.transpose(0, 1) # Dim = (batch, timesteps, 8)
        data = self.linear(data) # Dim = (batch, timesteps, channels)
        data = data.transpose(1, 2) # Dim = (batch, channels, timesteps)
        mask = self.activation(data) # Dim = (batch, channels, timesteps)

        estimates = input * mask
        return estimates
