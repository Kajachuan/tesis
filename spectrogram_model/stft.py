import torch
import torch.nn as nn

class STFT(nn.Module):
    """
    Calcula la STFT o ISTFT
    """
    def __init__(self, n_fft: int, hop: int) -> None:
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

    def forward(self, data: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Argumentos:
            data -- Señal de audio de dimensión (n_batch, n_channels, n_timesteps) si inverse=False
                    STFT de dimensión (n_batch, n_channels, n_bins, n_frames) si inverse=True
            inverse -- True si se calcula la ISTFT
        Retorna:
            STFT de dimensión (n_batch, n_channels, n_bins, n_frames) si inverse=False
            Señal de audio de dimensión (n_batch, n_channels, n_timesteps) si inverse=True
        """
        if not inverse:
            n_batch, n_channels, self.length = data.size()
            data = data.reshape(n_batch * n_channels, -1) # Dim: (n_batch * n_channels, n_timesteps)

            # Calcula la STFT
            data = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop,
                              window=self.window, center=True, normalized=False,
                              onesided=True, pad_mode='reflect', return_complex=False)
                   # Dim: (n_batch * n_channels, n_bins, n_frames, 2)

            _, n_bins, n_frames, __ = data.size()
            data = data.reshape(n_batch, n_channels, n_bins, n_frames, 2)
                   # Dim: (n_batch, n_channels, n_bins, n_frames, 2)
            real, imag = data[..., 0], data[..., 1]

            eps = 1e-8
            mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=eps))
            phase = torch.atan2(imag, real + eps)

            data = torch.stack((mag, phase), dim=-1)
            return data
        else:
            n_batch, n_channels, n_bins, n_frames, _ = data.size()
            data = data.reshape(n_batch * n_channels, n_bins, n_frames, 2)
                   # Dim: (n_batch * n_channels, n_bins, n_frames, 2)

            # Calculo la ISTFT
            data = torch.istft(data, n_fft=self.n_fft, hop_length=self.hop,
                               center=True, normalized=False, onesided=True,
                               return_complex=False, length=self.length)
                   # Dim: (n_batch * n_channels, n_timesteps)
            data = data.reshape(n_batch, n_channels, -1) # Dim: (n_batch, n_channels, n_timesteps)
            return data
