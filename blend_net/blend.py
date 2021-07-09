import torch
import torch.nn as nn
from utils.stft import STFT
from spectrogram_model.model import SpectrogramModel
from wave_model.model import WaveModel

class STFTConvLayer(nn.Module):
    def __init__(self, features: int, in_channels: int, out_channels: int = -1) -> None:
        """
        Argumentos:
            features -- Número de características de entrada (no del BatchNorm)
            in_channels -- Número de canales de entrada
            out_channels -- Número de canales de salida
        """
        super(STFTConvLayer, self).__init__()
        if out_channels == -1:
            out_channels = 2 * in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=(0,1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.batch_norm = nn.BatchNorm1d((features - 2) // 2)
        self.relu = nn.LeakyReLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Datos de entrada de dimensión (n_batch, in_channels, features, n_frames)

        Retorna:
            Resultado de dimensión (n_batch, out_channels, (features - 2) // 2, n_frames)
        """
        data = self.conv(data) # Dim = (n_batch, out_channels, features - 2, n_frames)
        data = self.pool(data) # Dim = (n_batch, out_channels, (features - 2) / 2, n_frames)
        data = data.transpose(1, 2) # Dim = (n_batch, (features - 2) / 2, out_channels, n_frames)
        data = data.reshape(data.size(0), data.size(1), -1) # Dim = (n_batch, (features - 2) / 2, out_channels * n_frames)
        data = self.batch_norm(data) # Dim = (n_batch, (features - 2) / 2, out_channels * n_frames)
        data = data.reshape(data.size(0), data.size(1), self.out_channels, -1) # Dim = (n_batch, (features - 2) / 2, out_channels, n_frames)
        data = data.transpose(1, 2) # Dim = (n_batch, out_channels, (features - 2) / 2, n_frames)
        data = self.relu(data) # Dim = (n_batch, out_channels, (features - 2) / 2, n_frames)
        return data

class WaveConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = -1) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada
            out_channels -- Número de canales de salida
        """
        super(WaveConvLayer, self).__init__()
        if out_channels == -1:
            out_channels = 2 * in_channels
        self.seq = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=3, padding=1),
                                 nn.BatchNorm1d(num_features=out_channels),
                                 nn.LeakyReLU())

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Datos de entrada de dimensión (n_batch, in_channels, timesteps)

        Retorna:
            Resultado de dimensión (n_batch, out_channels, timesteps)
        """
        return self.seq(data)

class BlendNet(nn.Module):
    """
    Modelo de mezcla de modelos de espectrograma y wave
    """
    def __init__(self, layers_spec: int, layers_wave: int, channels: int, nfft: int, hop: int, activation: str) -> None:
        """
        Argumentos:
            layers_spec -- Número de capas de la rama spec
            layers_wave -- Número de capas de la rama wave
            channels -- Número de canales de audio
            nfft -- Número de puntos para calcular la nfft
            hop -- Número de puntos de hop
            activation -- Función de activación a utilizar
        """
        super(BlendNet, self).__init__()
        self.channels = channels
        self.bins = nfft // 2 + 1
        blend = 2

        self.stft = STFT(nfft, hop)

        self.conv_stft = nn.Sequential(*([STFTConvLayer(features=self.bins, in_channels=blend * channels, out_channels=8)] +
                                         [STFTConvLayer(features=(self.bins - (2**i - 2)) // 2**(i-1), in_channels=2**(i+1))
                                          for i in range(2, layers_spec + 1)]))

        self.linear_stft = nn.Linear(in_features=(self.bins - (2**(layers_spec+1) - 2)) // 2**(layers_spec) * 2**(layers_spec+2),
                                     out_features=blend * self.bins * channels)

        self.conv_wave = nn.Sequential(*([WaveConvLayer(in_channels=(blend+1) * channels, out_channels=16)] +
                                         [WaveConvLayer(in_channels=2**(i+2)) for i in range(2, layers_wave + 1)]))

        self.linear_wave = nn.Linear(in_features=2**(layers_wave + 3), out_features=(blend + 1) * channels)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

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
        data = self.conv_stft(data) # Dim = (n_batch, 128, bins_out, n_frames)
        data = data.reshape(data.size(0), -1, data.size(-1)) # Dim = (n_batch, 128 * bins_out, n_frames)
        data = data.transpose(1, 2) # Dim = (n_batch, n_frames, 128 * bins_out)
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
        data = torch.stack([wave_stft, wave, blend_stft], dim=1) # Dim = (n_batch, 3, n_channels, timesteps)
        data = data.reshape(data.size(0), -1, data.size(-1)) # Dim = (n_batch, 3 * n_channels, timesteps)
        data = self.conv_wave(data) # Dim = (n_batch, 128, timesteps)
        data = data.transpose(1, 2) # Dim = (n_batch, timesteps, 128)
        data = self.linear_wave(data) # Dim = (n_batch, timesteps, n_channels * 3)
        data = data.reshape(data.size(0), data.size(1), self.channels, -1) # Dim = (n_batch, timesteps, n_channels, 3)
        data = data.transpose(1, 2) # Dim = (n_batch, n_channels, timesteps, 3)
        data = self.activation(data) # Dim = (n_batch, timesteps, n_channels, 3)
        data = data[..., 0] * wave_stft + data[..., 1] * wave + data[..., 2] * blend_stft
        return data
