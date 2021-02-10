import torch
import torch.nn as nn

class Spectrogram(nn.Module):
    """
    Calcula el espectrograma
    """
    def __init__(self, n_fft: int = 4096, hop: int = 1024, db_conversion: bool = True) -> None:
        """
        Argumentos:
            n_fft -- Número de bins de frecuencia
            hop -- Tamaño del hop
            db_conversion -- Si el espectrograma debe estar en dB
        """
        super(Spectrogram, self).__init__()

        self.n_fft = n_fft
        self.hop = hop
        self.db_conversion = db_conversion

        # Ventana de Hann
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            signal -- Señal de audio de dimensión (n_batch, n_channels, n_timesteps)

        Retorna:
            Espectrograma de dimensión ?
        """
        n_batch, n_channels, n_timesteps = signal.size()

        signal = signal.reshape(n_batch * n_channels, -1) # Dim: (n_batch * n_channels, n_timesteps)

        # Calculo la STFT
        stft = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, center=False, normalized=False,
                          onesided=True, pad_mode='reflect') # Dim: (n_batch, N, T, 2)

        stft = stft.contiguous().view(n_batch, n_channels, self.n_fft // 2 + 1, -1, 2)
        # Dim: (n_batch, n_channels, n_bins, n_frames, 2)

        stft = stft.pow(2).sum(-1) # Dim: (n_batch, n_channels, n_bins, n_frames)
        if self.db_conversion:
            stft = torch.log10(stft + 1e-8) # Dim: (n_batch, n_channels, n_bins, n_frames)

        return stft
