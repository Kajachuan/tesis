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

class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        '''
        Argumentos:
            d_model -- Tamaño de la dimensión de embedding
            heads -- Cantidad de heads para MultiheadAttention
        '''
        super(AttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        '''
        Argumentos:
            data -- Tensor de dimensión (batch, d_model, timesteps)
        Retorna:
            Tensor de dimensión (batch, d_model, timesteps)
        '''
        data = data.transpose(1, 2) # Dim: (batch, timesteps, d_model)
        skip = self.attn(data, data, data)[0]
        data = data + skip
        data = self.norm1(data)
        skip = self.linear2(self.relu(self.linear1(data)))
        data = data + skip
        data = self.norm2(data)
        data = data.transpose(1, 2) # Dim: (batch, d_model, timesteps)
        return data

class AttentionModel(nn.Module):
    def __init__(self, 
                 layers: int = 6, 
                 channels: int = 64, 
                 lstm_layers: int = 2, 
                 attn_layers: int = 2, 
                 heads: int = 4) -> None:
        '''
        Argumentos:
            layers -- Cantidad de capas (encoder/decoder)
            channels -- Canales de salida del primer encoder
            lstm_layers -- Cantidad de capas BLSTM
            attn_layers -- Cantidad de capas de atención
            heads -- Cantidad de heads de atención múltiple (válido para attn_layers > 0)
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

        self.embed = in_channels

        self.lstm = BLSTM(self.embed, lstm_layers) if lstm_layers else None

        self.attn = nn.Sequential(*[AttentionLayer(self.embed, heads) 
                                    for _ in range(attn_layers)]) if attn_layers else None

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
        if self.attn:
            x = x + pos_encoding(self.embed, x.size(-1), x.device)
            x = self.attn(x)
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x.size(-1))
            x = x + skip
            x = decode(x)
        
        x = julius.resample_frac(x, 2, 1)
        x = x * std + mean
        return x