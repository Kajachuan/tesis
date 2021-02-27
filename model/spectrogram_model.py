import torch
import torch.nn as nn
from model.batch_norm import BatchNorm
from model.blstm import BLSTM
from model.mask import Mask

class Model(nn.Module):
    """
    Modelo para separación de instrumentos
    """
    def __init__(self, n_channels: int, hidden_size: int, num_layers: int,
                 dropout: float, n_fft: int) -> None:
        """
        Argumentos:
            n_channels -- Número de canales de audio
            hidden_size -- Cantidad de unidades en cada capa BLSTM
            num_layers -- Cantidad de capas BLSTM
            dropout -- Dropout de las capas BLSTM
            n_fft -- Tamaño de la fft para el espectrograma
        """
        super(Model, self).__init__()

        n_bins = n_fft // 2 + 1
        self.batch_norm = BatchNorm(n_bins)
        self.blstm = BLSTM(n_channels * n_bins, hidden_size, num_layers, dropout)
        self.mask = Mask(n_bins, 2 * hidden_size, n_channels)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Espectrograma de dimensión (n_batch, n_channels, n_bins, n_frames)

        Retorna:
            Máscara, Estimación
        """

        mix = data
        data = 10 * torch.log10(data)
        data = self.batch_norm(data)
        data = self.blstm(data)
        mask = self.mask(data)

        estimates = mix * mask

        return mask, estimates
