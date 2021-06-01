import torch
import torch.nn as nn
from utils.stft import STFT
from spectrogram_model.model import SpectrogramModel
from wave_model.model import WaveModel

class BlendNet(nn.Module):
    """
    Modelo de mezcla de modelos de espectrograma y wave
    """
    def __init__(self, channels: int, nfft: int, hop: int, dropout: int) -> None:
        """
        Argumentos:
            channels -- Número de canales de audio
            nfft -- Número de puntos para calcular la nfft
            hop -- Número de puntos de hop
            dropout -- Probabilidad de dropout
        """
        super(BlendNet, self).__init__()
        self.channels = channels
        self.nfft = nfft
        self.bins = self.nfft // 2 + 1
        self.hop = hop
        hidden = 10
        layers = 1
        blend = 2

        self.stft = STFT(self.nfft, self.hop)

        self.lstm_stft = nn.LSTM(input_size=blend * self.bins * self.channels,
                                 hidden_size=hidden,
                                 num_layers=layers,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=True)
        self.linear_stft = nn.Linear(in_features=2 * hidden, out_features=blend * self.bins * self.channels)

        self.lstm_wave = nn.LSTM(input_size=blend * self.channels,
                                 hidden_size=hidden,
                                 num_layers=layers,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=True)
        self.linear_wave = nn.Linear(in_features=2 * hidden, out_features=blend * self.channels)

        self.linear_output = nn.Linear(in_features=blend * self.channels, out_features=blend * self.channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave_stft: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            wave_stft -- Audio del modelo de STFT de dimensión (n_batch, n_channels, timesteps)
            wave -- Audio del modelo de Wave de dimensión (n_batch, n_channels, timesteps)

        Retorna:
            Separación de dimensión (n_batch, n_channels, timesteps)
        """
        # Mezcla con STFT
        stft_stft = self.stft(wave_stft)
        mag_stft, phase_stft = stft_stft[..., 0], stft_stft[..., 1]

        stft_wave = self.stft(wave)
        mag_wave, phase_wave = stft_wave[..., 0], stft_wave[..., 1]

        data = torch.stack((mag_stft, mag_wave), dim=-1) # Dim = (n_batch, n_channels, n_bins, n_frames, 2)
        data = 10 * torch.log10(torch.clamp(data, min=1e-8)) # Dim = (n_batch, n_channels, n_bins, n_frames, 2)
        data = data.transpose(1, 3) # Dim = (n_batch, n_frames, n_bins, n_channels, 2)
        data = data.reshape(data.size(0), data.size(1), -1) # Dim = (n_batch, n_frames, n_bins * n_channels * 2)
        data = self.lstm_stft(data)[0] # Dim = (n_batch, n_frames, 2 * hidden_size)
        data = self.linear_stft(data) # Dim = (n_batch, n_frames, n_bins * n_channels * 2)
        data = self.sigmoid(data) # Dim = (n_batch, n_frames, n_bins * n_channels * 2)
        data = data.reshape(data.size(0), data.size(1), self.bins, self.channels, -1) # Dim = (n_batch, n_frames, n_bins, n_channels, 2)
        data = data.transpose(1, 3) # Dim = (n_batch, n_channels, n_frames, n_bins, 2)

        estim_stft = torch.stack((data[..., 0] * (mag_stft * torch.cos(phase_stft)) +
                                  data[..., 1] * (mag_wave * torch.cos(phase_wave)),
                                  data[..., 0] * (mag_stft * torch.sin(phase_stft)) +
                                  data[..., 1] * (mag_wave * torch.sin(phase_wave))), dim=-1)
        blend_stft = self.stft(estim_stft, inverse=True)

        # Mezcla con Wave
        data = torch.stack([wave_stft, wave], dim=-1) # Dim = (n_batch, n_channels, timesteps, 2)
        data = data.transpose(1, 2) # Dim = (n_batch, timesteps, n_channels, 2)
        data = data.reshape(data.size(0), data.size(1), -1) # Dim = (n_batch, timesteps, n_channels * 2)
        data = self.lstm_wave(data)[0] # Dim = (n_batch, timesteps, 2 * hidden_size)
        data = self.linear_wave(data) # Dim = (n_batch, timesteps, n_channels * 2)
        data = self.sigmoid(data) # Dim = (n_batch, timesteps, n_channels * 2)
        data = data.reshape(data.size(0), data.size(1), self.channels, -1) # Dim = (n_batch, timesteps, n_channels, 2)
        data = data.transpose(1, 2) # Dim = (n_batch, n_channels, timesteps, 2)
        blend_wave = data[..., 0] * wave_stft + data[..., 1] * wave

        # Mezclo todo
        data = torch.stack([blend_stft, blend_wave], dim=-1) # Dim = (n_batch, n_channels, timesteps, 2)
        data = data.transpose(1, 2) # Dim = (n_batch, timesteps, n_channels, 2)
        data = data.reshape(data.size(0), data.size(1), -1) # Dim = (n_batch, timesteps, n_channels * 2)
        data = self.linear_output(data) # Dim = (n_batch, timesteps, n_channels * 2)
        data = self.sigmoid(data) # Dim = (n_batch, timesteps, n_channels * 2)
        data = data.reshape(data.size(0), data.size(1), self.channels, -1) # Dim = (n_batch, timesteps, n_channels, 2)
        data = data.transpose(1, 2) # Dim = (n_batch, n_channels, timesteps, 2)
        data = data[..., 0] * blend_stft + data[..., 1] * blend_wave
        return data
