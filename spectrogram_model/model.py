import torch
import torch.nn as nn
from spectrogram_model.stft import STFT
from spectrogram_model.batch_norm import BatchNorm
from spectrogram_model.blstm import BLSTM
from spectrogram_model.mask import Mask
from typing import Tuple

class SpectrogramModel(nn.Module):
    """
    Modelo para separación de instrumentos
    """
    def __init__(self, n_channels: int, hidden_size: int, num_layers: int,
                 dropout: float, n_fft: int, hop: int) -> None:
        """
        Argumentos:
            n_channels -- Número de canales de audio
            hidden_size -- Cantidad de unidades en cada capa BLSTM
            num_layers -- Cantidad de capas BLSTM
            dropout -- Dropout de las capas BLSTM
            n_fft -- Tamaño de la fft para el espectrograma
            hop -- Tamaño del hop del espectrograma
        """
        super(SpectrogramModel, self).__init__()

        n_bins = n_fft // 2 + 1
        self.n_fft = n_fft
        self.hop = hop
        self.stft = STFT(n_fft, hop)
        self.batch_norm = BatchNorm(n_bins)
        self.blstm = BLSTM(n_channels * n_bins, hidden_size, num_layers, dropout)
        self.mask = Mask(n_bins, 2 * hidden_size, n_channels)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Argumentos:
            data -- Audio de dimensión (n_batch, n_channels, n_timesteps)

        Retorna:
            Máscara
            Estimación (magnitud del espectrograma)
            Estimación (wave)
        """

        stft = self.stft(data)
        mag, phase = stft[..., 0], stft[..., 1]
        mag_db = 10 * torch.log10(torch.clamp(mag, min=1e-8))
        data = self.batch_norm(mag_db)
        data = self.blstm(data)
        mask = self.mask(data)

        estim_mag = mag * mask
        estim_stft = torch.stack((estim_mag * torch.cos(phase),
                                  estim_mag * torch.sin(phase)), dim=-1)
        estimates = self.stft(estim_stft, inverse=True)

        return mask, estim_mag, estimates
