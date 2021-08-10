import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stft import STFT

def pos_encoding(embed_dim, length, device):
    pos_enc = torch.zeros(length, embed_dim, device=device)
    pos = torch.arange(0, length).unsqueeze(1)
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pos_enc[:, 0::2] = torch.sin(pos * div)
    pos_enc[:, 1::2] = torch.cos(pos * div)
    return pos_enc.unsqueeze(0)

def zero_pad(x, skip):
    diff = skip.size(-1) - x.size(-1)
    l_pad = diff // 2
    r_pad = diff - l_pad
    x = F.pad(x, (l_pad, r_pad), 'constant', 0)
    return x

class Encoder(nn.Module):
    def __init__(self, nfft: int, hop: int, heads: int, dropout: float,
                 in_channels: int, out_channels: int) -> None:
        super(Encoder, self).__init__()
        self.bins = nfft // 2 + 1
        self.stft = STFT(nfft, hop)
        self.attention = nn.MultiheadAttention(embed_dim=self.bins * in_channels, num_heads=heads, 
                                               dropout=dropout, batch_first=True)
        self.norm_att = nn.LayerNorm(normalized_shape=self.bins * in_channels)
        self.linear = nn.Linear(in_features=self.bins * in_channels, out_features=self.bins * in_channels)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.bins * in_channels)
        self.glu = nn.GLU(dim=1)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=8, stride=4)
        self.relu = nn.LeakyReLU()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        '''
        Argumentos:
            wave -- Tensor de dimensión (batch, in_channels, timesteps)
        '''
        # 1° Parte: Frecuencia
        stft = self.stft(wave)
        mag, phase = stft[..., 0], stft[..., 1]
        mag_db = 10 * torch.log10(torch.clamp(mag, min=1e-8)) # Dim: (batch, in_channels, bins, frames)
        mag_db = mag_db.transpose(1, 2) # Dim: (batch, bins, in_channels, frames)
        mag_db = mag_db.reshape(mag_db.size(0), -1, mag_db.size(-1)) # Dim: (batch, bins * in_channels, frames)
        mag_db = mag_db.transpose(1, 2) # Dim: (batch, frames, bins * in_channels)
        mag_db = mag_db + pos_encoding(mag_db.size(-1), mag_db.size(1), mag_db.device)

        mag = self.attention(mag_db, mag_db, mag_db)
        mag = mag + mag_db
        skip = self.norm_att(mag)
        mag = self.linear(skip)
        mag = mag + skip
        mag = self.norm_linear(mag)

        mag_db = mag.transpose(1, 2) # Dim: (batch, bins * in_channels, frames)
        mag_db = mag_db.reshape(mag_db.size(0), self.bins, -1, mag_db.size(-1)) # Dim: (batch, bins, in_channels, frames)
        mag_db = mag_db.transpose(1, 2) # Dim: (batch, in_channels, bins, frames)
        stft = torch.stack((mag_db * torch.cos(phase), mag_db * torch.sin(phase)), dim=-1)
        new_wave = self.stft(stft, inverse=True) # Dim: (batch, in_channels, timesteps)
        wave = torch.cat((wave, new_wave), dim=1) # Dim: (batch, 2 * in_channels, timesteps)
        wave = self.glu(wave) # Dim: (batch, in_channels, timesteps)

        # 2° Parte: Tiempo
        wave = self.conv(wave) # Dim: (batch, out_channels, timesteps)
        wave = self.relu(wave)
        return wave, mag

class Decoder(nn.Module):
    def __init__(self, nfft: int, hop: int, heads: int, dropout: float,
                 in_channels: int, out_channels: int) -> None:
        super(Decoder, self).__init__()
        self.bins = nfft // 2 + 1
        self.stft = STFT(nfft, hop)
        self.attention = nn.MultiheadAttention(embed_dim=self.bins * out_channels, num_heads=heads, 
                                               dropout=dropout, batch_first=True)
        self.norm_att = nn.LayerNorm(normalized_shape=self.bins * out_channels)
        self.linear = nn.Linear(in_features=self.bins * out_channels, out_features=self.bins * out_channels)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.bins * out_channels)
        self.glu = nn.GLU(dim=1)
        self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=8, stride=4)
        self.relu = nn.LeakyReLU()


    def forward(self, wave: torch.Tensor, mag_skip: torch.Tensor) -> torch.Tensor:
        '''
        Argumentos:
            wave -- Tensor proveniente del decoder anterior de dimensión (batch, in_channels, timesteps)
            mag_skip -- Tensor del STFT del encoder de dimensión (batch, frames, bins * out_channels)
        '''
        # 1° Parte: Tiempo
        wave = self.conv(wave) # Dim: (batch, out_channels, timesteps)
        wave = self.relu(wave)

        # 2° Parte: Frecuencia
        stft = self.stft(wave)
        mag, phase = stft[..., 0], stft[..., 1]
        mag_db = 10 * torch.log10(torch.clamp(mag, min=1e-8)) # Dim: (batch, out_channels, bins, frames)
        mag_db = mag_db.transpose(1, 2) # Dim: (batch, bins, out_channels, frames)
        mag_db = mag_db.reshape(mag_db.size(0), -1, mag_db.size(-1)) # Dim: (batch, bins * out_channels, frames)
        mag_db = mag_db.transpose(1, 2) # Dim: (batch, frames, bins * out_channels)
        mag_db = mag_db + pos_encoding(mag_db.size(-1), mag_db.size(1), mag_db.device)

        mag = self.attention(mag_skip, mag_skip, mag_db)
        mag = mag + mag_db
        skip = self.norm_att(mag)
        mag = self.linear(skip)
        mag = mag + skip
        mag = self.norm_linear(mag)

        mag_db = mag.transpose(1, 2) # Dim: (batch, bins * out_channels, frames)
        mag_db = mag_db.reshape(mag_db.size(0), self.bins, -1, mag_db.size(-1)) # Dim: (batch, bins, out_channels, frames)
        mag_db = mag_db.transpose(1, 2) # Dim: (batch, out_channels, bins, frames)
        stft = torch.stack((mag_db * torch.cos(phase), mag_db * torch.sin(phase)), dim=-1)
        new_wave = self.stft(stft, inverse=True) # Dim: (batch, out_channels, timesteps)
        wave = torch.cat((wave, new_wave), dim=1) # Dim: (batch, 2 * out_channels, timesteps)
        wave = self.glu(wave) # Dim: (batch, out_channels, timesteps)
        return wave

class AttentionModel(nn.Module):
    def __init__(self, layers: int, nfft: int, hop: int, dropout: float) -> None:
        super(AttentionModel, self).__init__()
        self.encoder = nn.ModuleList([Encoder(nfft=nfft, hop=hop, 
                                              heads=2**i if i < 3 else 8, dropout=dropout, 
                                              in_channels=2**i, out_channels=2**(i+1))
                                              for i in range(1, layers + 1)])
        self.decoder = nn.ModuleList([Decoder(nfft=nfft, hop=hop,
                                              heads=2**i if i < 3 else 8, dropout=dropout,
                                              in_channels=2**(i+1), out_channels=2**i)
                                              for i in range(layers, 0, -1)])

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        x = mix
        mono = mix.mean(dim=1, keepdim=True)
        mean = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)

        x = (x - mean) / (1e-8 + std)

        skips = []
        for encode in self.encoder:
            x, mag_skip = encode(x)
            skips.append((x, mag_skip))
        
        for decode in self.decoder:
            wave_skip, mag_skip = skips.pop(-1)
            x = zero_pad(x, wave_skip)
            x = x + wave_skip
            x = decode(x, mag_skip)

        x = x * std + mean
        return x