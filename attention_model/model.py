# El modelo básico es exactamente igual al de demucs: https://github.com/facebookresearch/demucs

import math
import julius
import torch
import torch.nn as nn
from utils.stft import STFT
from attention_model.utils import *

class BLSTM(nn.Module):
    def __init__(self, hidden_size: int, layers: int) -> None:
        '''
        Argumentos:
            hidden_size -- Cantidad de unidades BLSTM
            layers -- Cantidad de capas BLSTM
        '''
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        '''
        Argumentos:
            data -- Tensor de dimensión (batch, hidden_size, timesteps)
        Retorna:
            Tensor de dimensión (batch, hidden_size, timesteps)
        '''
        data = data.transpose(1, 2) # Dim: (batch, timesteps, hidden_size)
        data = self.lstm(data)[0] # Dim: (batch, timesteps, 2 * hidden_size)
        data = self.linear(data) # Dim: (batch, timesteps, hidden_size)
        data = data.transpose(1, 2) # Dim: (batch, hidden_size, timesteps)
        return data

class AttentionModel(nn.Module):
    def __init__(self,
                 layers: int = 6,
                 channels: int = 64,
                 lstm_layers: int = 2) -> None:
        '''
        Argumentos:
            layers -- Cantidad de capas (encoder/decoder)
            channels -- Canales de salida del primer encoder
            kernel_size -- Tamaño del kernel
        '''
        super(AttentionModel, self).__init__()
        self.layers = layers
        self.kernel_size = 8
        self.stride = 4
        self.context = 3
        rescale = 0.1
        in_channels = 2

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        glu = nn.GLU(dim=1)
        relu = nn.ReLU()
        for index in range(layers):
            encode = [nn.Conv1d(in_channels, channels, self.kernel_size, self.stride), relu,
                      nn.Conv1d(channels, 2 * channels, 1), glu]
            self.encoder.append(nn.Sequential(*encode))

            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 2

            decode = [nn.Conv1d(channels, 2 * channels, self.context), glu,
                      nn.ConvTranspose1d(channels, out_channels, self.kernel_size, self.stride)]
            if index > 0:
                decode.append(relu)
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = 2 * channels

        channels = in_channels

        self.lstm = BLSTM(channels, lstm_layers) if lstm_layers else None

        # self.attention = 

        for sub in self.modules():
            if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
                std = sub.weight.std().detach()
                scale = (std / rescale)**0.5
                sub.weight.data /= scale
                if sub.bias is not None:
                    sub.bias.data /= scale

    def calculate_length(self, length: int) -> int:
        length *= 2
        for _ in range(self.layers):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.layers):
            length = (length - 1) * self.stride + self.kernel_size

        length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        '''
        Argumentos:
            mix -- Tensor de dimensión (batch, 2, timesteps)
        Retorna:
            Tensor de dimensión (batch, 2, timesteps')
        '''
        if not self.training:
            target_length = self.calculate_length(mix.size(-1))
            mix = pad_tensor(mix, target_length)

        x = mix

        mono = mix.mean(dim=1, keepdim=True)
        mean = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)

        x = (x - mean) / (1e-8 + std)
        x = julius.resample_frac(x, 1, 2)
        
        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
        if self.lstm:
            x = self.lstm(x)
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x.size(-1))
            x = x + skip
            x = decode(x)
        
        x = julius.resample_frac(x, 2, 1)
        x = x * std + mean
        return x