import torch
import torch.nn as nn
from model.spectrogram import Spectrogram
from model.batch_norm import BatchNorm
from model.blstm import BLSTM
from model.embedding import Embedding

class HierarchicalModel(nn.Module):
    """
    Modelo jerárquico para separación de instrumentos
    """
    def __init__(self, n_fft: int = 4096, hop: int = 1024, n_channels: int, hidden_size: int,
                 num_layers: int, dropout: float, n_sources: int) -> None:
        """
        Argumentos:
            n_fft -- Tamaño de la fft para el espectrograma
            hop -- Tamaño del hop del espectrograma
            n_channels -- Número de canales de audio
            hidden_size -- Cantidad de unidades en cada capa BLSTM
            num_layers -- Cantidad de capas BLSTM
            dropout -- Dropout de las capas BLSTM
            n_sources -- Número de instrumentos
        """
        super(HierarchicalModel, self).__init__()

        n_bins = n_fft // 2 + 1
        self.spectrogram = Spectrogram(n_fft, hop)
        self.batch_norm = BatchNorm(n_bins)
        self.blstm = BLSTM(n_channels * n_bins, hidden_size, num_layers, dropout)
        self.embedding = Embedding(n_bins, 2 * hidden_size, n_sources, n_channels)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Audio de dimensión (n_batch, n_channels, n_timesteps)

        Retorna:
            Máscara, Estimación
        """
        mix = data

        data = self.spectrogram(mix)
        data = self.batch_norm(data)
        data = self.blstm(data)
        mask = self.embedding(data)

        estimates = mix.unsqueeze(-1) * mask
        return mask, estimates
