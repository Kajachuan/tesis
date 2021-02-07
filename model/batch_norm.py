import torch.nn as nn

class BatchNorm(nn.Module):
    """
    Capa de normalización por lote (batch)

    Argumentos:
        num_features {int} -- Argumento num_features de BatchNorm1d
        feature_dim {int} -- Dimensión que será normalizada
        **kwargs {dict} -- Argumentos de BatchNorm1d
    """

    def __init__(self, num_features=1, feature_dim=2, **kwargs):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.batch_norm = nn.BatchNorm1d(num_features, **kwargs)

    def forward(self, data):
        """
        Argumentos:
            data {torch.Tensor} -- Entrada que será normalizada. Dim: (B, T, F, C)

        Retorna:
            torch.Tensor -- Entrada normalizada

        Referencia:
            B: Tamaño del lote (batch)
            T: Número de bins de tiempo
            F: Número de bins de frecuencia
            C: Número de canales (mono, stereo, etc.)
        """
        data = data.transpose(self.feature_dim, 1) # Dim: (B, F, T, C)
        shape = data.shape
        new_shape = (shape[0], self.num_features, -1)

        data = data.reshape(new_shape) # Dim: (B, num_features, -1)
        data = self.batch_norm(data) # Dim: (B, num_features, -1)
        data = data.reshape(shape) # Dim: (B, F, T, C)
        data = data.transpose(self.feature_dim, 1) # Dim: (B, T, F, C)
        return data
