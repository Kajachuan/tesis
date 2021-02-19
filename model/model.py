import torch
import torch.nn as nn
from model.stft import STFT
from model.batch_norm import BatchNorm
from model.blstm import BLSTM
from model.mask import Mask

class Model(nn.Module):
    """
    Modelo para separación de instrumentos
    """
    def __init__(self, n_channels: int, hidden_size: int, num_layers: int,
                 dropout: float, n_fft: int = 4096, hop: int = 1024) -> None:
        """
        Argumentos:
            n_channels -- Número de canales de audio
            hidden_size -- Cantidad de unidades en cada capa BLSTM
            num_layers -- Cantidad de capas BLSTM
            dropout -- Dropout de las capas BLSTM
            n_fft -- Tamaño de la fft para el espectrograma
            hop -- Tamaño del hop del espectrograma
        """
        super(Model, self).__init__()

        n_bins = n_fft // 2 + 1
        self.n_fft = n_fft
        self.hop = hop
        self.stft = STFT(n_fft, hop)
        self.batch_norm = BatchNorm(n_bins)
        self.blstm = BLSTM(n_channels * n_bins, hidden_size, num_layers, dropout)
        self.mask = Mask(n_bins, 2 * hidden_size, n_channels)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Audio de dimensión (n_batch, n_channels, n_timesteps)

        Retorna:
            Máscara, Estimación
        """

        stft = self.stft(data)
        mag, phase = stft[0, ...], stft[1, ...]
        data = self.batch_norm(mag)
        data = self.blstm(data)
        mask = self.mask(data)

        estim_mag = mag * mask
        estim_stft = torch.stack((estim_mag * torch.cos(phase),
                                  estim_mag * torch.sin(phase)), dim=-1)
        n_batch, n_channels, n_bins, n_frames, _ = estim_stft.size()
        estim_stft = estim_stft.reshape(n_batch * n_channels, n_bins, n_frames, -1)
        estimates = torch.istft(estim_stft, n_fft=self.n_fft, hop_length=self.hop, center=True,
                                normalized=False, onesided=True, return_complex=False)

        return mask, estimates
