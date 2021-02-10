import torch.nn as nn

class BLSTM(nn.Module):
    """
    Crea una pila de capas BLSTM
    """
    def __init__(self, num_features: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Argumentos:
            num_features -- Número de features
            hidden_size -- Cantidad de unidades en cada capa (tamaño de las capas ocultas)
            num_layers -- Número de capas BLSTM
            dropout -- Dropout entre capas
        """
        super(BLSTM, self).__init__()
        self.blstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True,
                             dropout=dropout, bidirectional=True)

        # Inicialización
        # for name, param in self.blstm.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)

    def forward(self, data: torch.Tensor):
        """
        Argumentos:
            data -- Entrada de dimensión (n_batch, n_channels, n_bins, n_frames)

        Retorna:
            torch.Tensor -- Salida de dimensión (n_batch, n_frames, hidden_size)
        """
        data = data.transpose(1, 3) # Dim: (n_batch, n_frames, n_bins, n_channels)
        shape = data.shape
        data = data.reshape(shape[0], shape[1], -1) # Dim: (n_batch, n_frames, *)
        self.blstm.flatten_parameters()
        data = self.blstm(data)[0] # Dim: (n_batch, n_frames, hidden_size)
        return data
