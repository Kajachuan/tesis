import torch
import torch.nn as nn
from wave_model.blocks import *

class WaveModel(nn.Module):
    """
    Modelo para separación de instrumentos utilizando wav
    """
    def __init__(self, n_channels: int, layers: int, filters: int, size_down: int, size_up: int) -> None:
        """
        Argumentos:
            n_channels -- Número de canales de audio
            layers -- Número de capas
            filters -- Número de filtros extra por capa
            size_down -- Tamaño del filtro de downsampling
            size_up -- Tamaño del filtro de upsampling
        """
        super(WaveModel, self).__init__()

        self.layers = layers
        self.downsampling = nn.ModuleList([DownsamplingBlock(n_channels, filters, size_down)])
        self.downsampling.extend([DownsamplingBlock(filters * (l-1), filters * l, size_down)
                                  for l in range(2, layers + 1)])

        self.middle = MiddleBlock(layers * filters, (layers + 1) * filters, size_down)

        self.upsampling = nn.ModuleList([UpsamplingBlock(filters * (l+1), filters * l, size_up)
                                         for l in range(1, layers + 1)])

        self.output = OutputBlock(filters, 2, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bypass = {}
        concat_input = input
        current = input
        for i in range(self.layers):
            current, concat = self.downsampling[i](current)
            bypass[i] = concat

        current = self.middle(current)
        for i in range(self.layers - 1, -1, -1):
            current = self.upsampling[i](current, bypass[i])

        current = self.output(current, concat_input)
        return current
