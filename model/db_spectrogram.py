import torch
import torch.nn as nn
import numpy as np

class DBSpectrogram(nn.Module):
    """
    Recibe un espectrograma y lo convierte en decibeles

    Argumentos:
        data {torch.Tensor} -- El espectrograma
    Retorna:
        torch.Tensor -- Energía en decibeles en cada bin
    """

    def forward(self, data):
        data = data ** 2
        data = 10.0 * (torch.log10(torch.clamp(data, min=1e-8))) # El valor minimo no puede ser 0
        return data
