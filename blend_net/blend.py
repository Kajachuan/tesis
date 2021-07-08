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
        self.conv = nn.Conv2d(in_channels=2 * channels, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.rnn = nn.GRU(input_size=self.bins // 2 * 8, hidden_size=self.bins // 2 * 4, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.unpool = nn.MaxUnpool2d(2)
        self.deconv = nn.ConvTranspose2d(in_channels=8, out_channels=2 * channels, kernel_size=3)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, wave_stft: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
        stft_stft = self.stft(wave_stft)
        mag_stft, phase_stft = stft_stft[..., 0], stft_stft[..., 1]

        stft_wave = self.stft(wave)
        mag_wave, phase_wave = stft_wave[..., 0], stft_wave[..., 1]

        mag = torch.stack((mag_stft, mag_wave), dim=1) # Dim = (batch, 2, channels, bins, frames)
        data = 10 * torch.log10(torch.clamp(mag, min=1e-8)) # Dim = (batch, 2, channels, bins, frames)
        data = data.reshape(data.size(0), -1, data.size(-2), data.size(-1)) # Dim = (batch, 2 * channels, bins, frames)
        data = self.conv(data) # Dim = (batch, 8, bins, frames)
        out_size = data.size()
        data, idx = self.pool(data) # Dim = (batch, 8, bins // 2, frames // 2)
        data = data.reshape(data.size(0), -1, data.size(-1)) # Dim = (batch, bins // 2 * 8, frames // 2)
        data = data.transpose(1, 2) # Dim = (batch, frames // 2, bins // 2 * 8)
        data = self.rnn(data) # Dim = (batch, frames // 2, bins // 2 * 8)
        data = data.transpose(1, 2) # Dim = (batch, bins // 2 * 8, frames // 2)
        data = data.reshape(data.size(0), -1, self.bins // 2, data.size(-1)) # Dim = (batch, 8, bins // 2, frames // 2)
        data = self.unpool(data, idx, out_size) # Dim = (batch, 8, bins, frames)
        data = self.deconv(data) # Dim = (batch, 2 * channels, bins, frames)
        data = data.reshape(data.size(0), 2, -1, data.size(-2), data.size(-1)) # Dim = (batch, 2, channels, bins, frames)
        data = self.activation(data)

        estim_stft = torch.stack((data[:, 0, ...] * (mag_stft * torch.cos(phase_stft)) +
                                  data[:, 1, ...] * (mag_wave * torch.cos(phase_wave)),
                                  data[:, 0, ...] * (mag_stft * torch.sin(phase_stft)) +
                                  data[:, 1, ...] * (mag_wave * torch.sin(phase_wave))), dim=-1)
        data = self.stft(estim_stft, inverse=True)
        return data
