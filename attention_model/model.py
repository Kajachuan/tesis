import math
from typing import Tuple
import torch
import torch.nn as nn

def pos_encoding(embed_dim, length, device):
    pos_enc = torch.zeros(length, embed_dim, device=device)
    pos = torch.arange(0, length).unsqueeze(1)
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pos_enc[:, 0::2] = torch.sin(pos * div)
    pos_enc[:, 1::2] = torch.cos(pos * div)
    return pos_enc.unsqueeze(0)

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, groups: int = 1) -> None:
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups)
    
    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_real(real) - self.conv_imag(imag), \
               self.conv_real(imag) + self.conv_imag(real)

class ComplexReLU(nn.Module):
    def __init__(self) -> None:
        super(ComplexReLU, self).__init__()
        self.relu_real = nn.ReLU()
        self.relu_imag = nn.ReLU()

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.relu_real(real), self.relu_imag(imag)

class ComplexDropout(nn.Module):
    def __init__(self, p: float) -> None:
        super(ComplexDropout, self).__init__()
        self.dropout_real = nn.Dropout(p)
        self.dropout_imag = nn.Dropout(p)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dropout_real(real), self.dropout_imag(imag)

class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super(ComplexLayerNorm, self).__init__()
        self.norm_real = nn.LayerNorm(normalized_shape)
        self.norm_imag = nn.LayerNorm(normalized_shape)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.norm_real(real), self.norm_imag(imag)

class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(ComplexLinear, self).__init__()
        self.linear_real = nn.Linear(in_features, out_features)
        self.linear_imag = nn.Linear(in_features, out_features)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.linear_real(real) - self.linear_imag(imag), \
               self.linear_real(imag) + self.linear_imag(real)

class ComplexMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, 
                 dropout: float = 0.0, batch_first: bool = False) -> None:
        super(ComplexMultiheadAttention, self).__init__()
        self.mh = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mh(real, real, real)[0] + self.mh(real, imag, imag)[0] - \
               self.mh(imag, real, imag)[0] - self.mh(imag, imag, real)[0], \
               self.mh(real, real, imag)[0] + self.mh(real, imag, real)[0] + \
               self.mh(imag, real, real)[0] - self.mh(imag, imag, imag)[0]

class AttentionModel(nn.Module):
    def __init__(self, nfft: int, hop: int) -> None:
        super(AttentionModel, self).__init__()
        self.nfft = nfft
        self.hop = hop
        self.bins = nfft // 2 + 1

        dropout = 0.3
        heads = 2
        d_model = heads * self.bins

        self.window = nn.Parameter(torch.hann_window(nfft), requires_grad=False)
        self.attn = ComplexMultiheadAttention(embed_dim=d_model, num_heads=heads, 
                                              dropout=dropout, batch_first=True)
        self.conv1 = ComplexConv2d(in_channels=2, out_channels=heads,
                                   kernel_size=1, groups=2)
        self.conv2 = ComplexConv2d(in_channels=heads, out_channels=2,
                                   kernel_size=1, groups=2)
        self.linear1 = ComplexLinear(d_model, 4 * d_model)
        self.linear2 = ComplexLinear(4 * d_model, d_model)
        self.norm1 = ComplexLayerNorm(d_model)
        self.norm2 = ComplexLayerNorm(d_model)
        self.dropout1 = ComplexDropout(dropout)
        self.dropout2 = ComplexDropout(dropout)
        self.dropout3 = ComplexDropout(dropout)
        self.relu = ComplexReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        '''
        Argumentos:
            mix -- Tensor de dimensión (batch, channels, timesteps)
        Retorna:
            Tensor de dimensión (batch, channels, timesteps)
        '''
        length = mix.size(-1)

        # Calculo STFT
        stft = mix.reshape(-1, length) # Dim: (batch * 2, timesteps)
        stft = torch.stft(stft, n_fft=self.nfft, 
                          hop_length=self.hop, window=self.window, 
                          onesided=True, return_complex=True) # Dim: (batch * 2, bins, frames)
        stft = stft.reshape(-1, 2, self.bins, stft.size(-1)) # Dim: (batch, 2, bins, frames)

        # stft = stft.transpose(1, 3) # Dim: (batch, frames, bins, channels)
        # stft = stft.reshape(stft.size(0), stft.size(1), -1) # Dim: (batch, frames, bins * channels)

        # Parte real e imaginaria
        real, imag = torch.real(stft), torch.imag(stft)

        # Convolución 1
        new_real, new_imag = self.conv1(real, imag) # Dim: (batch, 8, bins, frames)
        new_real = new_real.transpose(1, 3).reshape(stft.size(0), stft.size(-1), -1) # Dim: (batch, frames, bins * 8)
        new_imag = new_imag.transpose(1, 3).reshape(stft.size(0), stft.size(-1), -1)

        # Positional encoding
        encoding = pos_encoding(new_real.size(-1), new_real.size(1), new_real.device)
        new_real = new_real + encoding
        new_imag = new_imag + encoding

        # Atención compleja
        new_real, new_imag = self.attn(new_real, new_imag)
        new_real, new_imag = self.norm1(*self.dropout1(new_real, new_imag))

        # Feed Forward complejo
        new_real, new_imag = self.dropout2(*self.relu(*self.linear1(new_real, new_imag)))
        new_real, new_imag = self.norm2(*self.dropout3(*self.linear2(new_real, new_imag)))

        # Convolución 2
        new_real = new_real.reshape(new_real.size(0), new_real.size(1), self.bins, -1).transpose(1, 3) # Dim: (batch, 8, bins, frames)
        new_imag = new_imag.reshape(new_imag.size(0), new_imag.size(1), self.bins, -1).transpose(1, 3)
        new_real, new_imag = self.conv2(new_real, new_imag) # Dim: (batch, 2, bins, frames)

        # Máscara
        real = real * self.sigmoid(new_real)
        imag = imag * self.sigmoid(new_imag)
        
        # Calculo la STFT inversa
        stft = real + 1j * imag
        stft = stft.reshape(-1, self.bins, stft.size(-1)) # Dim: (batch * 2, bins, frames)

        estim = torch.istft(stft, n_fft=self.nfft, hop_length=self.hop, 
                            window=self.window, onesided=True, return_complex=False, 
                            length=length) # Dim: (batch * 2, timesteps)
        estim = estim.reshape(-1, 2, estim.size(-1)) # Dim: (batch, 2, timesteps)
        return estim, torch.stack((torch.real(stft), torch.imag(stft)), dim=-1)