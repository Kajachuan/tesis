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
                              onesided=True, pad_mode='reflect', return_complex=True)
                   # Dim: (n_batch * n_channels, n_bins, n_frames)

            _, n_bins, n_frames = data.size()
            data = data.reshape(n_batch, n_channels, n_bins, n_frames) # Dim: (n_batch, n_channels, n_bins, n_frames)

            mag = torch.abs(data)
            phase = torch.angle(data)

            data = torch.stack((mag, phase), dim=-1)
            return data
        else:
            n_batch, n_channels, n_bins, n_frames = data.size()
            data = data.reshape(n_batch * n_channels, n_bins, n_frames)
                   # Dim: (n_batch * n_channels, n_bins, n_frames)

            # Calculo la ISTFT
            data = torch.istft(data, n_fft=self.n_fft, hop_length=self.hop,
                               center=True, normalized=False, onesided=True,
                               return_complex=False, length=self.length)
                   # Dim: (n_batch * n_channels, n_timesteps)
            data = data.reshape(n_batch, n_channels, -1) # Dim: (n_batch, n_channels, n_timesteps)
            return data
