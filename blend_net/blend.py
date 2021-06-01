import torch
import torch.nn as nn
from utils.stft import STFT
from spectrogram_model.model import SpectrogramModel
from wave_model.model import WaveModel

class BlendNet(nn.Module):
    """
    Modelo de mezcla de modelos de espectrograma y wave
    """
    def __init__(self, channels: int, nfft: int, hop: int) -> None:
        """
        Argumentos:
            channels -- Número de canales de audio
            nfft -- Número de puntos para calcular la nfft
            hop -- Número de puntos de hop
        """
        super(BlendNet, self).__init__()
        self.channels = channels
        self.nfft = nfft
        self.bins = self.nfft // 2 + 1
        self.hop = hop
        blend = 2

        self.stft = STFT(self.nfft, self.hop)

        self.conv_stft = nn.Sequential(
                             nn.Conv2d(in_channels=blend * self.channels,
                                       out_channels=8,
                                       kernel_size=3,
                                       padding=(0, 1)),
                             nn.MaxPool2d(kernel_size=(2, 1)),
                             nn.Conv2d(in_channels=8,
                                       out_channels=16,
                                       kernel_size=3,
                                       padding=(0, 1)),
                             nn.MaxPool2d(kernel_size=(2, 1)),
                             nn.Conv2d(in_channels=16,
                                       out_channels=32,
                                       kernel_size=3,
                                       padding=(0, 1)),
                             nn.MaxPool2d(kernel_size=(2, 1))
                         ) # h_out = (h_in - 14) / 8, w_out = w_in

        self.linear_stft = nn.Linear(in_features=8 * (self.bins - 14),
                                     out_features=blend * self.bins * self.channels) # 64 * (in - 14) / 8 = 8 * (in - 14)

        self.lstm_wave = nn.LSTM(input_size=(blend + 1) * self.channels,
                                 hidden_size=16,
                                 num_layers=1,
                                 batch_first=True)
        self.linear_wave = nn.Linear(in_features=16, out_features=(blend + 1) * self.channels)

        self.activation = nn.Softmax(dim=-1)

    def forward(self, wave_stft: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            wave_stft -- Audio del modelo de STFT de dimensión (n_batch, n_channels, timesteps)
            wave -- Audio del modelo de Wave de dimensión (n_batch, n_channels, timesteps)

        Retorna:
            Separación de dimensión (n_batch, n_channels, timesteps)
        """
        # Mezcla con STFT
        stft_stft = self.stft(wave_stft)
        mag_stft, phase_stft = stft_stft[..., 0], stft_stft[..., 1]

        stft_wave = self.stft(wave)
        mag_wave, phase_wave = stft_wave[..., 0], stft_wave[..., 1]

        data = torch.stack((mag_stft, mag_wave), dim=1) # Dim = (n_batch, 2, n_channels, n_bins, n_frames)
        data = 10 * torch.log10(torch.clamp(data, min=1e-8)) # Dim = (n_batch, 2, n_channels, n_bins, n_frames)
        data = data.reshape(data.size(0), -1, data.size(-2), data.size(-1)) # Dim = (n_batch, 2 * n_channels, n_bins, n_frames)
        data = self.conv_stft(data) # Dim = (n_batch, 64, bins_out, n_frames)
        data = data.reshape(data.size(0), -1, data.size(-1)) # Dim = (n_batch, 64 * bins_out, n_frames)
        data = data.transpose(1, 2) # Dim = (n_batch, n_frames, 64 * bins_out)
        data = self.linear_stft(data) # Dim = (n_batch, n_frames, 2 * n_bins * n_channels)
        data = data.reshape(data.size(0), data.size(1), self.bins, self.channels, -1) # Dim = (n_batch, n_frames, n_bins, n_channels, 2)
        data = data.transpose(1, 3) # Dim = (n_batch, n_channels, n_bins, n_frames, 2)
        data = self.activation(data) # Dim = (n_batch, n_channels, n_bins, n_frames, 2)

        estim_stft = torch.stack((data[..., 0] * (mag_stft * torch.cos(phase_stft)) +
                                  data[..., 1] * (mag_wave * torch.cos(phase_wave)),
                                  data[..., 0] * (mag_stft * torch.sin(phase_stft)) +
                                  data[..., 1] * (mag_wave * torch.sin(phase_wave))), dim=-1)
        blend_stft = self.stft(estim_stft, inverse=True)

        # Mezcla con Wave
        data = torch.stack([wave_stft, wave, blend_stft], dim=-1) # Dim = (n_batch, n_channels, timesteps, 3)
        data = data.transpose(1, 2) # Dim = (n_batch, timesteps, n_channels, 3)
        data = data.reshape(data.size(0), data.size(1), -1) # Dim = (n_batch, timesteps, n_channels * 3)
        data = self.lstm_wave(data)[0] # Dim = (n_batch, timesteps, hidden_size)
        data = self.linear_wave(data) # Dim = (n_batch, timesteps, n_channels * 3)
        data = data.reshape(data.size(0), data.size(1), self.channels, -1) # Dim = (n_batch, timesteps, n_channels, 3)
        data = data.transpose(1, 2) # Dim = (n_batch, n_channels, timesteps, 3)
        data = self.activation(data) # Dim = (n_batch, timesteps, n_channels, 3)
        data = data[..., 0] * wave_stft + data[..., 1] * wave + data[..., 2] * blend_stft
        return data
