import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    Módulo de inferencia de la máscara
    """
    def __init__(self, n_bins: int, hidden_size: int, n_sources: int, n_channels: int) -> None:
        """
        Argumentos:
            n_bins -- Número de bins de frecuencia
            hidden_size -- Número de unidades en cada capa BLSTM (anterior)
            n_sources -- Cantidad de instrumentos
            n_channels -- Número de canales del audio
        """
        super(Embedding, self).__init__()
        self.linear = nn.Linear(hidden_size, n_bins * n_channels * n_sources)
        self.n_bins = n_bins
        self.n_channels = n_channels
        self.n_sources = n_sources

        # Inicialización
        # for name, param in self.linear.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Entrada de dimensión (n_batch, n_frames, hidden_size)

        Retorna:
            Una máscara de dimensión (n_batch, n_frames, n_bins, n_channels, n_sources)
        """
        shape = data.shape
        data = self.linear(data) # Dim: (n_batch, n_frames, n_bins * n_channels * n_sources)
        data = data.reshape(shape[0], shape[1], self.n_bins, self.n_channels, self.n_sources)
               # Dim: (n_batch, n_frames, n_bins, n_channels, n_sources)
        data = torch.sigmoid(data)
        return data
