import torch
import torch.nn as nn
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

        self.padding = nn.ZeroPad2d((0, self.nfft, 0, 0))
        self.conv = nn.Conv1d(in_channels=self.channels,
                              out_channels=self.channels * self.bins,
                              kernel_size=self.nfft,
                              stride=self.hop)
        self.deconv = nn.ConvTranspose1d(in_channels=2 * self.channels * self.bins,
                                         out_channels=self.channels,
                                         kernel_size=self.nfft,
                                         stride=self.hop)
        self.linear = nn.Linear(in_features=3 * self.channels, out_features=self.channels)

    def forward(self, stft: torch.Tensor, wave_stft: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            stft -- STFT de dicho modelo de dimensión (n_batch, n_channels, n_bins, n_frames)
            wave_stft -- Audio recuperado de la STFT de dimensión (n_batch, n_channels, timesteps)
            wave -- Audio del modelo de Wave de dimensión (n_batch, n_channels, timesteps)

        Retorna:
            Separación de dimensión (n_batch, n_channels, timesteps)
        """
        timesteps = wave.size(-1)

        data = self.padding(wave)                                           # Dim: (n_batch, n_channels, timesteps + nfft)
        data = self.conv(data)                                              # Dim: (n_batch, n_channels * n_bins, n_frames)
        data = data.reshape(data.size(0), self.channels, -1, data.size(-1)) # Dim: (n_batch, n_channels, n_bins, n_frames)
        data = torch.stack([data, stft], dim=1)                             # Dim: (n_batch, 2, n_channels, n_bins, n_frames)
        data = data.reshape(data.size(0), -1, data.size(-1))                # Dim: (n_batch, 2 * n_channels * n_bins, n_frames)
        data = self.deconv(data)                                            # Dim: (n_batch, n_channels, out_length)
        data = data[..., :timesteps]                                        # Dim: (n_batch, n_channels, timesteps)

        data = torch.stack([data, wave_stft, wave], dim=-1)                 # Dim: (n_batch, n_channels, timesteps, 3)
        data = data.transpose(1, 2)                                         # Dim: (n_batch, timesteps, n_channels, 3)
        data = data.reshape(data.size(0), data.size(1), -1)                 # Dim: (n_batch, timesteps, 3 * n_channels)
        data = self.linear(data)                                            # Dim: (n_batch, timesteps, n_channels)
        data = data.transpose(1, 2)                                         # Dim: (n_batch, n_channels, timesteps)
        return data
