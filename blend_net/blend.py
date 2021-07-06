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
                             batch_first=True, dropout=0.3, bidirectional=True)
        self.linear = nn.Linear(8, channels)
        self.activation = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # stft = self.stft(data)
        # mag, phase = stft[..., 0], stft[..., 1]
        # mag_db = 10 * torch.log10(torch.clamp(mag, min=1e-8)) # Dim: (n_batch, n_channels, n_bins, n_frames)
        # Dim = (n_batch, n_channels, timesteps)
        mag = input * 2
        mag_db = 10 * torch.log10(torch.clamp(mag, min=1e-8))
        data = mag_db.transpose(1, 2) # Dim: (n_batch, timesteps, n_channels)
        # data = data.reshape(data.size(0), data.size(1), -1) # Dim: (n_batch, n_frames, n_bins * n_channels)
        self.blstm.flatten_parameters()
        data = self.blstm(data)[0] # Dim: (n_batch, timesteps, hidden)
        data = self.linear(data) # Dim: (n_batch, timesteps, n_channels)
        # data = data.reshape(data.size(0), data.size(1), self.bins, self.channels) # Dim: (n_batch, n_frames, n_bins, n_channels)
        data = data.transpose(1, 2) # Dim: (n_batch, n_channels, timesteps)
        mask = self.activation(data)
        estim = mag * mask

        # estim = mag * mask
        # estim_stft = torch.stack((estim_mag * torch.cos(phase),
        #                           estim_mag * torch.sin(phase)), dim=-1)
        # estimates = self.stft(estim_stft, inverse=True)
        return estim
