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

        self.stft = STFT(self.nfft, self.hop)

        self.linear_mag = nn.Linear(in_features=2 * self.bins * self.channels, out_features=self.bins * self.channels)
        self.prelu_mag = nn.PReLU(num_parameters=self.channels)

        self.linear_wave = nn.Linear(in_features=2 * self.channels, out_features=self.channels)

        self.linear_output = nn.Linear(in_features=4 * self.channels, out_features=self.channels)
        self.tanh = nn.Tanh()

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
        mag_stft = 10 * torch.log10(torch.clamp(mag_stft, min=1e-8))

        stft_wave = self.stft(wave)
        mag_wave, phase_wave = stft_wave[..., 0], stft_wave[..., 1]
        mag_wave = 10 * torch.log10(torch.clamp(mag_wave, min=1e-8))

        mag = torch.stack([mag_stft, mag_wave], dim=-1) # Dim = (n_batch, n_channels, n_bins, n_frames, 2)
        mag = mag.transpose(1, 3) # Dim = (n_batch, n_frames, n_bins, n_channels, 2)
        mag = mag.reshape(mag.size(0), mag.size(1), -1) # Dim = (n_batch, n_frames, n_bins * n_channels * 2)
        mag = self.linear_mag(mag) # Dim = (n_batch, n_frames, n_bins * n_channels)
        mag = mag.reshape(mag.size(0), mag.size(1), -1, self.channels) # Dim = (n_batch, n_frames, n_bins, n_channels)
        mag = mag.transpose(1, 3) # Dim = (n_batch, n_channels, n_bins, n_frames)
        mag = self.prelu_mag(mag) # Dim = (n_batch, n_channels, n_bins, n_frames)

        phase = (phase_stft + phase_wave) / 2

        estim_stft = torch.stack((mag * torch.cos(phase),
                                  mag * torch.sin(phase)), dim=-1)
        blend_stft = self.stft(estim_stft, inverse=True)

        # Mezcla con Wave
        data = torch.stack([wave_stft, wave], dim=-1) # Dim = (n_batch, n_channels, timesteps, 2)
        data = data.transpose(1, 2) # Dim = (n_batch, timesteps, n_channels, 2)
        data = data.reshape(data.size(0), data.size(1), -1) # Dim = (n_batch, timesteps, n_channels * 2)
        data = self.linear_wave(data) # Dim = (n_batch, timesteps, n_channels)
        data = data.transpose(1, 2) # Dim = (n_batch, n_channels, timesteps)
        blend_wave = self.tanh(data) # Dim = (n_batch, n_channels, timesteps)

        # Mezclo todo
        data = torch.stack([wave_stft, wave, blend_stft, blend_wave], dim=-1) # Dim = (n_batch, n_channels, timesteps, 4)
        data = data.transpose(1, 2) # Dim = (n_batch, timesteps, n_channels, 4)
        data = data.reshape(data.size(0), data.size(1), -1) # Dim = (n_batch, timesteps, n_channels * 4)
        data = self.linear_output(data) # Dim = (n_batch, timesteps, n_channels)
        data = data.transpose(1, 2) # Dim = (n_batch, n_channels, timesteps)
        data = self.tanh(data) # Dim = (n_batch, n_channels, timesteps)
        return data
