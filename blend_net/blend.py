import torch
import torch.nn as nn
from utils.stft import STFT

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
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Datos de entrada de dimensión (n_batch, in_channels, timesteps)

        Retorna:
            Resultado de dimensión (n_batch, out_channels, timesteps)
        """
        data = self.conv(data) # Dim = (n_batch, out_channels, timesteps)
        data = self.batch_norm(data) # Dim = (n_batch, out_channels, timesteps)
        data = self.relu(data) # Dim = (n_batch, out_channels, timesteps)
        return data

class BlendNet(nn.Module):
    """
    Modelo de mezcla de modelos de espectrograma y wave
    """
    def __init__(self, blend: int, layers: int, channels: int, nfft: int, hop: int) -> None:
        """
        Argumentos:
            layers -- Número de capas
            channels -- Número de canales de audio
            nfft -- Número de puntos para calcular la nfft
            hop -- Número de puntos de hop
            activation -- Función de activación a utilizar
        """
        super(BlendNet, self).__init__()
        self.blend = blend
        self.channels = channels
        self.bins = nfft // 2 + 1

        self.stft = STFT(nfft, hop)

        self.conv_stft = nn.Sequential(*([STFTConvLayer(features=self.bins, in_channels=blend * channels, out_channels=8)] +
                                         [STFTConvLayer(features=(self.bins - (2**i - 2)) // 2**(i-1), in_channels=2**(i+1))
                                          for i in range(2, layers + 1)]))

        self.linear_stft = nn.Linear(in_features=(self.bins - (2**(layers+1) - 2)) // 2**(layers) * 2**(layers+2),
                                     out_features=blend * self.bins * channels)

        self.conv_wave = nn.Sequential(*([WaveConvLayer(in_channels=(blend+1) * channels, out_channels=8)] +
                                         [WaveConvLayer(in_channels=2**(i+1)) for i in range(2, layers + 1)]))

        self.linear_wave = nn.Linear(in_features=2**(layers + 2), out_features=(blend + 1) * channels)

        self.activation = nn.Tanh()

    def forward(self, *args) -> torch.Tensor:
        """
        Argumentos:
            *args -- Tensores a mezclar

        Retorna:
            Separación de dimensión (n_batch, n_channels, timesteps)
        """
        assert len(args) == self.blend

        # Mezcla con STFT
        mag, phase = [], []
        for tensor in args:
            stft = self.stft(tensor)
            mag.append(stft[..., 0])
            phase.append(stft[..., 1])

        data = torch.stack(mag, dim=1) # Dim = (batch, blend, channels, bins, frames)
        data = 10 * torch.log10(torch.clamp(data, min=1e-8))
        data = data.reshape(data.size(0), -1, data.size(-2), data.size(-1)) # Dim = (batch, blend * channels, bins, frames)
        data = self.conv_stft(data) # Dim = (batch, out_channels, bins_out, frames)
        data = data.reshape(data.size(0), -1, data.size(-1)) # Dim = (batch, out_channels * bins_out, frames)
        data = data.transpose(1, 2) # Dim = (batch, frames, out_channels * bins_out)
        data = self.linear_stft(data) # Dim = (batch, frames, blend * bins * channels)
        data = data.reshape(data.size(0), data.size(1), self.bins, self.channels, -1) # Dim = (batch, frames, bins, channels, blend)
        data = data.transpose(1, 3) # Dim = (batch, channels, bins, frames, blend)
        data = self.activation(data) # Dim = (batch, channels, bins, frames, blend)

        real = data[..., 0] * mag[0] * torch.cos(phase[0]) 
        imag = data[..., 0] * mag[0] * torch.sin(phase[0])
        for i in range(1, self.blend):
            real += data[..., i] * mag[i] * torch.cos(phase[i])
            imag += data[..., i] * mag[i] * torch.sin(phase[i])

        estim_stft = torch.stack((real, imag), dim=-1)
        blend_stft = self.stft(estim_stft, inverse=True)

        # Mezcla con Wave
        args = list(args) + [blend_stft]
        data = torch.stack(args, dim=1) # Dim = (batch, blend + 1, channels, timesteps)
        data = data.reshape(data.size(0), -1, data.size(-1)) # Dim = (batch, (blend + 1) * channels, timesteps)
        data = self.conv_wave(data) # Dim = (batch, channels_out, timesteps)
        data = data.transpose(1, 2) # Dim = (batch, timesteps, channels_out)
        data = self.linear_wave(data) # Dim = (batch, timesteps, channels * (blend + 1))
        data = data.reshape(data.size(0), data.size(1), self.channels, -1) # Dim = (batch, timesteps, channels, blend + 1)
        data = data.transpose(1, 2) # Dim = (batch, channels, timesteps, blend + 1)
        data = self.activation(data) # Dim = (batch, timesteps, channels, blend + 1)

        sum = data[..., 0] * args[0]
        for i in range(i, self.blend):
            sum += data[..., i] * args[i]

        return sum