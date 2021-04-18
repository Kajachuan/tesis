import torch
import torch.nn as nn
from typing import Tuple

def downsample(x: torch.Tensor) -> torch.Tensor:
    """
    Decimación por un factor de 2
    """
    return x[..., ::2]

def upsample(x: torch.Tensor) -> torch.Tensor:
    """
    Interpolación por un factor de 2
    """
    n_batch, channels, timesteps = x.size()
    out = torch.zeros(n_batch, channels, 2 * timesteps - 1)
    out[..., ::2] = x
    out[..., 1::2] = (x[..., :-1] + x[..., 1:]) / 2
    return out

class DownsamplingBlock(nn.Module):
    """
    Bloque de downsampling
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada
            out_channels -- Número de canales de salida
            kernel_size -- Tamaño del kernel
        """
        super(DownsamplingBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Argumentos:
            data -- Entrada del bloque de dimensión (n_batch, in_channels, in_length)

        Retorna:
            Salida del bloque de dimensión (n_batch, out_channels, out_length / 2)
            Salida de la convolución de dimensión (n_batch, out_channels, out_length)
        """
        data = self.conv(data)
        data = nn.LeakyReLU(negative_slope=0.2, inplace=True)(data)
        down = downsample(data)
        return down, data

class MiddleBlock(nn.Module):
    """
    Bloque ubicado entre el final de los bloques de downsampling
    y el inicio de los bloques de upsampling
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada
            out_channels -- Número de canales de salida
            kernel_size -- Tamaño del kernel
        """
        super(MiddleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Entrada del bloque de dimensión (n_batch, in_channels, in_length)

        Retorna:
            Salida de la convolución de dimensión (n_batch, out_channels, out_length)
        """
        data = self.conv(data)
        data = nn.LeakyReLU(negative_slope=0.2, inplace=True)(data)
        return data

class UpsamplingBlock(nn.Module):
    """
    Bloque de upsampling
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada (del bloque, no de la convolución)
            out_channels -- Número de canales de salida
            kernel_size -- Tamaño del kernel
        """
        super(UpsamplingBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels + out_channels, out_channels, kernel_size)

    def forward(self, data: torch.Tensor, concat: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Entrada del bloque de dimensión (n_batch, in_channels, in_length)
            concat -- Datos para concatenar de dimensión (n_batch, out_channels, in_length')

        Retorna:
            Salida de la convolución de dimensión (n_batch, out_channels, out_length)
        """
        data = upsample(data)

        diff = concat.size(-1) - data.size(-1)
        l_pad = diff // 2
        r_pad = diff - l_pad
        data = nn.ZeroPad2d((l_pad, r_pad, 0, 0))(data.unsqueeze(0))[0]

        data = torch.cat([data, concat], dim=1)
        data = self.conv(data)
        data = nn.LeakyReLU(negative_slope=0.2, inplace=True)(data)
        return data

class OutputBlock(nn.Module):
    """
    Bloque final de salida
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        Argumentos:
            in_channels -- Número de canales de entrada (del bloque, no de la convolución)
            out_channels -- Número de canales de salida
            kernel_size -- Tamaño del kernel
        """
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels + out_channels, out_channels, kernel_size)

    def forward(self, data: torch.Tensor, concat: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Entrada del bloque de dimensión (n_batch, in_channels, in_length)
            concat -- Datos para concatenar de dimensión (n_batch, out_channels, in_length')

        Retorna:
            Salida de la convolución de dimensión (n_batch, out_channels, out_length)
        """
        diff = concat.size(-1) - data.size(-1)
        l_pad = diff // 2
        r_pad = diff - l_pad
        data = nn.ZeroPad2d((l_pad, r_pad, 0, 0))(data.unsqueeze(0))[0]

        data = torch.cat([data, concat], dim=1)
        data = self.conv(data)
        data = nn.Tanh()(data)
        return data
