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

        dropout = 0.3
        d_model = 2 * self.bins

        self.window = nn.Parameter(torch.hann_window(nfft), requires_grad=False)
        self.mh = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, 
                                        dropout=dropout, batch_first=True)
        self.linear1_real = nn.Linear(d_model, 4 * d_model)
        self.linear1_imag = nn.Linear(d_model, 4 * d_model)
        self.linear2_real = nn.Linear(4 * d_model, d_model)
        self.linear2_imag = nn.Linear(4 * d_model, d_model)
        self.norm1_real = nn.LayerNorm(d_model)
        self.norm1_imag = nn.LayerNorm(d_model)
        self.norm2_real = nn.LayerNorm(d_model)
        self.norm2_imag = nn.LayerNorm(d_model)
        self.dropout1_real = nn.Dropout(dropout)
        self.dropout1_imag = nn.Dropout(dropout)
        self.dropout2_real = nn.Dropout(dropout)
        self.dropout2_imag = nn.Dropout(dropout)
        self.dropout3_real = nn.Dropout(dropout)
        self.dropout3_imag = nn.Dropout(dropout)
        self.relu = nn.ReLU()
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
        real = real + encoding
        imag = imag + encoding

        # Atención compleja
        new_real = self.mh(real, real, real)[0] + self.mh(real, imag, imag)[0] - \
                   self.mh(imag, real, imag)[0] - self.mh(imag, imag, real)[0]
        new_imag = self.mh(real, real, imag)[0] + self.mh(real, imag, real)[0] + \
                   self.mh(imag, real, real)[0] - self.mh(imag, imag, imag)[0]

        # Sumo con la entrada
        new_real = self.norm1_real(real + self.dropout1_real(new_real))
        new_imag = self.norm1_imag(imag + self.dropout1_imag(new_imag))

        # Feed Forward complejo
        other_real, other_imag = self.dropout2_real(self.relu(self.linear1_real(new_real) - \
                                                              self.linear1_imag(new_imag))), \
                                 self.dropout2_imag(self.relu(self.linear1_real(new_imag) + \
                                                              self.linear1_imag(new_real)))
        new_real, new_imag = new_real + self.dropout3_real(self.linear2_real(other_real) - \
                                                           self.linear2_imag(other_imag)), \
                             new_imag + self.dropout3_imag(self.linear2_real(other_imag) + \
                                                           self.linear2_imag(other_real))
        new_real, new_imag = self.norm2_real(new_real), self.norm2_imag(new_imag)

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
        return estim, torch.stack((torch.real(stft), torch.imag(stft)), dim=-1)