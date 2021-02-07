import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    Módulo de inferencia de la máscara

    Argumentos:
        num_features {int} -- Número de features a ser mapeados por cada frame
        hidden_size {int} -- Tamaño de la salida del BLSTM
        embedding_size {int} -- Dimensionalidad del embedding
        num_audio_channels {int} -- Número de canales del audio
    """
    def __init__(self, num_features, hidden_size, embedding_size, num_audio_channels=1):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(hidden_size, num_features * num_audio_channels * embedding_size)
        self.num_features = num_features
        self.num_audio_channels = num_audio_channels
        self.embedding_size = embedding_size
        self.dim_to_embed = [-1]

        # Inicialización
        for name, param in self.linear.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param) # Revisar si se podría cambiar

    def forward(self, data): # Revisar esto
        """
        Argumentos:
            data {torch.Tensor} -- Salida del BLSTM de dimensión (B, T, H)

        Retorna:
            torch.Tensor -- Una máscara de dimensión (B, T, NF, C, S)

        Referencia:
            B: Tamaño del lote (batch)
            T: Número de bins de tiempo
            NF: Número de features
            C: Número de canales (mono, stereo, etc.)
            S: Número de fuentes (instrumentos)
        """
        shape = list(data.shape)
        _dims = []
        for _dim in self.dim_to_embed:
            if _dim == -1:
                _dim = len(shape) - 1
            data = data.transpose(_dim, -1)
            _dims.append(_dim)

        shape = [v for i, v in enumerate(shape) if i not in _dims]

        shape = tuple(shape)
        data = data.reshape(shape + (-1,))
        data = self.linear(data)

        shape = shape + (self.num_features, self.num_audio_channels, self.embedding_size,)
        data = data.reshape(shape)

        data = torch.sigmoid(data)
        return data
