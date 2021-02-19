import torch
import torch.nn as nn

class STFT(nn.Module):
    """
    Calcula la STFT de la señal
    """
    def __init__(self, n_fft: int = 4096, hop: int = 1024) -> None:
        """
        Argumentos:
            n_fft -- Tamaño de la FFT
            hop -- Tamaño del hop
        """
        super(STFT, self).__init__()

        self.n_fft = n_fft
        self.hop = hop

        # Ventana de Hann
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            signal -- Señal de audio de dimensión (n_batch, n_channels, n_timesteps)

        Retorna:
            STFT de dimensión (mag + phase, n_batch, n_channels, n_bins, n_frames)
        """
        n_batch, n_channels, n_timesteps = signal.size()

        signal = signal.reshape(n_batch * n_channels, -1) # Dim: (n_batch * n_channels, n_timesteps)

        # Calcula la STFT
        signal = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop,
                            window=self.window, center=True, normalized=False,
                            onesided=True, pad_mode='reflect', return_complex=False)
                 # Dim: (n_batch * n_channels, n_bins, n_frames, 2)

        signal *= torch.sqrt(1.0 / (self.window.sum() ** 2))

        _, n_bins, n_frames, __ = signal.size()
        signal = signal.reshape(n_batch, n_channels, n_bins, n_frames, 2) # Dim: (n_batch, n_channels, n_bins, n_frames, 2)
        real, imag = signal[..., 0], signal[..., 1]

        eps = 1e-8
        real[real.abs() < eps] = eps
        imag[imag.abs() < eps] = eps

        mag = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag, real)

        return torch.stack((mag, phase))
