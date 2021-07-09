import torch
import torch.nn as nn
from typing import Tuple
from utils.stft import STFT

class ConvLayer1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = -1) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada
            out_channels -- Número de canales de salida
        """
        super(ConvLayer1D, self).__init__()
        if out_channels == -1:
            out_channels = 2 * in_channels
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, return_indices=True)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """
        Argumentos:
            data -- Datos de entrada de dimensión (batch, in_channels, height, width)
        Retorna:
            Tupla con:
                - Resultado de dimensión (batch, out_channels, (features-2)//2, (width-2)//2)
                - Índices del max pooling
                - Tamaño original para hacer el upsampling
        """
        data = self.conv(data) # Dim = (batch, out_channels, height-2, width-2)
        size = data.size()
        data = self.batch_norm(data)
        data = self.relu(data)
        data, idx = self.pool(data) # Dim = (batch, out_channels, (height-2)//2, (width-2)//2)
        return data, idx, size

class DeconvLayer1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = -1) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada
            out_channels -- Número de canales de salida
        """
        super(DeconvLayer1D, self).__init__()
        if out_channels == -1:
            out_channels = in_channels // 2
        self.unpool = nn.MaxUnpool1d(kernel_size=2)
        self.deconv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, data: torch.Tensor, idx: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Datos de entrada de dimensión (batch, in_channels, (height-2)//2, (width-2)//2)
            idx -- Índices obtenidos con el max pooling
            size -- Tamaño resultante del upsampling
        Retorna:
            Tupla con:
                - Resultado de dimensión (batch, out_channels, height, width)
        """
        data = self.unpool(data, idx, size) # Dim = (batch, in_channels, height-2, width-2)
        data = self.deconv(data) # Dim = (batch, out_channels, height, width)
        data = self.batch_norm(data)
        data = self.relu(data)
        return data

class ConvLayer2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = -1) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada
            out_channels -- Número de canales de salida
        """
        super(ConvLayer2D, self).__init__()
        if out_channels == -1:
            out_channels = 2 * in_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """
        Argumentos:
            data -- Datos de entrada de dimensión (batch, in_channels, height, width)
        Retorna:
            Tupla con:
                - Resultado de dimensión (batch, out_channels, (features-2)//2, (width-2)//2)
                - Índices del max pooling
                - Tamaño original para hacer el upsampling
        """
        data = self.conv(data) # Dim = (batch, out_channels, height-2, width-2)
        size = data.size()
        data = self.batch_norm(data)
        data = self.relu(data)
        data, idx = self.pool(data) # Dim = (batch, out_channels, (height-2)//2, (width-2)//2)
        return data, idx, size

class DeconvLayer2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = -1) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada
            out_channels -- Número de canales de salida
        """
        super(DeconvLayer2D, self).__init__()
        if out_channels == -1:
            out_channels = in_channels // 2
        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, data: torch.Tensor, idx: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Datos de entrada de dimensión (batch, in_channels, (height-2)//2, (width-2)//2)
            idx -- Índices obtenidos con el max pooling
            size -- Tamaño resultante del upsampling
        Retorna:
            Tupla con:
                - Resultado de dimensión (batch, out_channels, height, width)
        """
        data = self.unpool(data, idx, size) # Dim = (batch, in_channels, height-2, width-2)
        data = self.deconv(data) # Dim = (batch, out_channels, height, width)
        data = self.batch_norm(data)
        data = self.relu(data)
        return data

class BlendNet(nn.Module):
    def __init__(self, layers: int, channels: int, nfft: int, hop: int) -> None:
        super(BlendNet, self).__init__()

        self.bins = nfft // 2 + 1
        self.channels = channels
        self.stft = STFT(nfft, hop)

        self.convs2d = nn.ModuleList([ConvLayer2D(in_channels=2 * channels, out_channels=8)] +
                                     [ConvLayer2D(in_channels=2**(i+1)) for i in range(2, layers+1)])

        self.deconvs2d = nn.ModuleList([DeconvLayer2D(in_channels=2**(i+2)) for i in range(layers, 1, -1)] +
                                       [DeconvLayer2D(in_channels=8, out_channels=2 * channels)])

        self.convs1d = nn.ModuleList([ConvLayer2D(in_channels=3 * channels, out_channels=8)] +
                                     [ConvLayer2D(in_channels=2**(i+1)) for i in range(2, layers+1)])

        self.deconvs1d = nn.ModuleList([DeconvLayer2D(in_channels=2**(i+2)) for i in range(layers, 1, -1)] +
                                       [DeconvLayer2D(in_channels=8, out_channels=3 * channels)])

        self.activation = nn.Sigmoid()

    def forward(self, wave_stft: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
        # Dominio de frecuencia
        blend = torch.stack((wave_stft, wave), dim=1) # Dim = (batch, 2, channels, timesteps)
        blend = blend.reshape(blend.size(0), -1, blend.size(-1)) # Dim = (batch, 2 * channels, timesteps)
        stft = self.stft(blend) # Dim = (batch, 2 * channels, bins, frames, 2)
        mag, phase_stft, phase_wave = stft[..., 0], stft[:, :2, ..., 1], stft[:, 2:, ..., 1]
        data = 10 * torch.log10(torch.clamp(mag, min=1e-8))

        bypass = {}
        for i in range(len(self.convs2d)):
            data, idx, size = self.convs2d[i](data)
            bypass[i] = (idx, size)

        for i in range(len(self.deconvs2d)):
            idx, size = bypass[len(self.deconvs2d) - 1 - i]
            data = self.deconvs2d[i](data, idx, size)

        mask = self.activation(data)
        estim_mag = mag * mask
        mag_stft, mag_wave = mag[:, :2, ...], estim_mag[:, 2:, ...]

        estim_stft = torch.stack((mag_stft * torch.cos(phase_stft) + mag_wave * torch.cos(phase_wave),
                                  mag_stft * torch.sin(phase_stft) + mag_wave * torch.sin(phase_wave)), dim=-1)
        blend_stft = self.stft(estim_stft, inverse=True) # Dim = (batch, channels, timesteps)

        # Dominio de tiempo
        blend = torch.stack((wave_stft, wave, blend_stft), dim=1) # Dim = (batch, 3, channels, timesteps)
        blend = blend.reshape(blend.size(0), -1, blend.size(-1)) # Dim = (batch, 3 * channels, timesteps)
        data = blend

        bypass = {}
        for i in range(len(self.convs1d)):
            data, idx, size = self.convs1d[i](data)
            bypass[i] = (idx, size)

        for i in range(len(self.deconvs1d)):
            idx, size = bypass[len(self.deconvs1d) - 1 - i]
            data = self.deconvs1d[i](data, idx, size)

        mask = self.activation(data) # Dim = (batch, 3 * channels, timesteps)
        data = blend * mask # Dim = (batch, 3 * channels, timesteps)
        data = data.reshape(data.size(0), 3, self.channels, -1) # Dim = (batch, 3, channels, timesteps)
        data = data.sum(1) # Dim = (batch, channels, timesteps)
        return data
