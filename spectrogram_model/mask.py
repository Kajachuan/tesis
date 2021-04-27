import torch
import torch.nn as nn

class Mask(nn.Module):
    """
    Módulo de inferencia de la máscara
    """
    def __init__(self, n_bins: int, hidden_size: int, n_channels: int) -> None:
        """
        Argumentos:
            n_bins -- Número de bins de frecuencia
            hidden_size -- Número de unidades en cada capa BLSTM (anterior)
            n_channels -- Número de canales del audio
        """
        super(Mask, self).__init__()
        self.linear = nn.Linear(hidden_size, n_bins * n_channels)
        self.n_bins = n_bins
        self.n_channels = n_channels

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Entrada de dimensión (n_batch, n_frames, hidden_size)

        Retorna:
            Una máscara de dimensión (n_batch, n_channels, n_bins, n_frames)
        """
        n_batch, n_frames, _ = data.size()
        data = self.linear(data) # Dim: (n_batch, n_frames, n_bins * n_channels)
        data = data.reshape(n_batch, n_frames, self.n_bins, self.n_channels)
               # Dim: (n_batch, n_frames, n_bins, n_channels)
        data = data.transpose(1, 3) # Dim: (n_batch, n_channels, n_bins, n_frames)
        data = torch.sigmoid(data)
        return data
