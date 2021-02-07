import torch.nn as nn

class BLSTM(nn.Module):
    """
    Crea varias capas BLSTM

    Argumentos:
        num_features {int} -- Número de features
        hidden_size {int} -- Cantidad de unidades en cada capa (tamaño de las capas ocultas)
        num_layers {int} -- Número de capas BLSTM
        dropout {float} -- Dropout entre capas
    """
    def __init__(self, num_features, hidden_size, num_layers, dropout):
        super(BLSTM, self).__init__()
        self.blstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True,
                             dropout=dropout, bidirectional=True)

        # Inicialización
        for name, param in self.blstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param) # Revisar si se podría cambiar

        # Revisar que hace esto
        for names in self.blstm._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(self.blstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def forward(self, data):
        """
        Argumentos:
            data {torch.Tensor} -- Entrada de dimensión (B, T, F, C)

        Retorna:
            torch.Tensor -- Salida de dimensión (B, T, H)

        Referencia:
            B: Tamaño del lote (batch)
            T: Número de bins de tiempo
            F: Número de bins de frecuencia
            C: Número de canales (mono, stereo, etc.)
            H: Cantidad de unidades (hidden size)
        """
        shape = data.shape
        data = data.reshape(shape[0], shape[1], -1) # Dim: (B, T, -1)
        self.blstm.flatten_parameters()
        data = self.blstm(data)[0] # Dim: (B, T, H)
        return data
