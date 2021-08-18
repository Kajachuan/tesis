import math
import torch
import torch.nn as nn

def pos_encoding(embed_dim, length, device):
    pos_enc = torch.zeros(length, embed_dim, device=device)
    pos = torch.arange(0, length).unsqueeze(1)
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pos_enc[:, 0::2] = torch.sin(pos * div)
    pos_enc[:, 1::2] = torch.cos(pos * div)
    return pos_enc.unsqueeze(0)

class AttentionModel(nn.Module):
    def __init__(self, nfft: int, hop: int) -> None:
        super(AttentionModel, self).__init__()
        self.nfft = nfft
        self.hop = hop
        self.bins = nfft // 2 + 1
        self.window = nn.Parameter(torch.hann_window(nfft), requires_grad=False)
        self.mh = nn.MultiheadAttention(embed_dim=2 * self.bins, num_heads=2, batch_first=True)
        self.linear1_real = nn.Linear(2 * self.bins, 4 * self.bins)
        self.linear1_imag = nn.Linear(2 * self.bins, 4 * self.bins)
        self.relu = nn.ReLU()
        self.linear2_real = nn.Linear(4 * self.bins, 2 * self.bins)
        self.linear2_imag = nn.Linear(4 * self.bins, 2 * self.bins)
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
        stft = mix.reshape(-1, length) # Dim: (batch * channels, timesteps)
        stft = torch.stft(stft, n_fft=self.nfft, 
                          hop_length=self.hop, window=self.window, 
                          onesided=True, return_complex=True) # Dim: (batch * channels, bins, frames)
        stft = stft.reshape(-1, 2, self.bins, stft.size(-1)) # Dim: (batch, channels, bins, frames)
        stft = stft.transpose(1, 3) # Dim: (batch, frames, bins, channels)
        stft = stft.reshape(stft.size(0), stft.size(1), -1) # Dim: (batch, frames, bins * channels)

        # Parte real e imaginaria
        real, imag = torch.real(stft), torch.imag(stft)

        # Positional encoding
        encoding = pos_encoding(stft.size(-1), stft.size(1), stft.device)
        del stft
        real = real + encoding
        imag = imag + encoding

        # Atención compleja
        new_real = self.mh(real, real, real)[0] + self.mh(real, imag, imag)[0] - \
                   self.mh(imag, real, imag)[0] - self.mh(imag, imag, real)[0]
        new_imag = self.mh(real, real, imag)[0] + self.mh(real, imag, real)[0] + \
                   self.mh(imag, real, real)[0] - self.mh(imag, imag, imag)[0]

        # Feed Forward complejo
        new_real, new_imag = self.relu(self.linear1_real(new_real) - self.linear1_imag(new_imag)), \
                             self.relu(self.linear1_real(new_imag) + self.linear1_imag(new_real))
        new_real, new_imag = self.linear2_real(new_real) - self.linear2_imag(new_imag), \
                             self.linear2_real(new_imag) + self.linear2_imag(new_real)

        # Máscara
        real = real * self.sigmoid(new_real)
        imag = imag * self.sigmoid(new_imag)
        
        # Calculo la STFT inversa
        stft = real + 1j * imag
        stft = stft.reshape(-1, stft.size(1), self.bins, 2) # Dim: (batch, frames, bins, channels)
        stft = stft.transpose(1, 3) # Dim: (batch, channels, bins, frames)
        stft = stft.reshape(-1, self.bins, stft.size(-1)) # Dim: (batch * channels, bins, frames)

        estim = torch.istft(stft, n_fft=self.nfft, hop_length=self.hop, 
                            window=self.window, onesided=True, return_complex=False, 
                            length=length) # Dim: (batch * channels, timesteps)
        estim = estim.reshape(-1, 2, estim.size(-1))
        return estim