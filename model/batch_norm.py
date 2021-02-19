import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    """
    Capa de normalización por lote (batch)
    """
    def __init__(self, num_features: int, **kwargs) -> None:
        """
        Argumentos:
            **kwargs {dict} -- Argumentos de BatchNorm1d
        """
        super(BatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, **kwargs)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data {torch.Tensor} -- Magnitud de STFT será normalizada de dimensión
                                   (n_batch, n_channels, n_bins, n_frames)

        Retorna:
            Entrada normalizada de dimensión (n_batch, n_channels, n_bins, n_frames)
        """

        data = data.transpose(1, 2) # Dim: (n_batch, n_bins, n_channels, n_frames)
        shape = data.shape

        data = data.reshape(shape[0], shape[1], -1) # Dim: (n_batch, n_bins, *)
        data = self.batch_norm(data) # Dim: (n_batch, n_bins, *)
        data = data.reshape(shape) # Dim: (n_batch, n_bins, n_channels, n_frames)
        data = data.transpose(1, 2) # Dim: (n_batch, n_channels, n_bins, n_frames)
        return data
