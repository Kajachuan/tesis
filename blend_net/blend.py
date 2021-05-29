import torch
import torch.nn as nn
from spectrogram_model.model import SpectrogramModel
from wave_model.model import WaveModel

class BlendNet(nn.Module):
    """
    Modelo de mezcla de modelos de espectrograma y wave
    """
    def __init__(self, stft_path: str, wave_path: str, device: torch.device) -> None:
        """
        Argumentos:
            stft_path -- Ruta del checkpoint del modelo de STFT
            wave_path -- Ruta del checkpoint del modelo de Wave
            device -- Device de PyTorch
        """
        super(BlendNet, self).__init__()

        stft_state = torch.load(f"{stft_path}/best_checkpoint", map_location=device)
        wave_state = torch.load(f"{wave_path}/best_checkpoint", map_location=device)

        stft_args = stft_state["args"]
        self.channels = stft_args[0]
        self.nfft = stft_args[-2]
        self.bins = self.nfft // 2 + 1
        self.hop = stft_args[-1]

        self.padding = nn.ZeroPad2d((0, self.nfft, 0, 0))
        self.conv = nn.Conv1d(in_channels=self.channels,
                              out_channels=self.channels * self.bins,
                              kernel_size=self.nfft,
                              stride=self.hop)
        self.deconv = nn.ConvTranspose1d(in_channels=2 * self.channels * self.bins,
                                         out_channels=self.channels,
                                         kernel_size=self.nfft,
                                         stride=self.hop)

        self.stft_model = SpectrogramModel(*stft_args).to(device)
        self.stft_model.load_state_dict(stft_state["state_dict"])
        self.stft_model.eval()

        self.wave_model = WaveModel(*wave_state["args"]).to(device)
        self.wave_model.load_state_dict(wave_state["state_dict"])
        self.wave_model.eval()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Argumentos:
            data -- Señales de audio de dimensión (n_batch, n_channels, timesteps)

        Retorna:
            Separación de dimensión (n_batch, n_channels, timesteps)
        """
        with torch.no_grad():
            _, stft, _ = self.stft_model(data)
            wave = self.wave_model(data)

        timesteps = data.size(-1)

        data = self.padding(wave)                                           # Dim: (n_batch, n_channels, timesteps + nfft)
        print(data.shape)
        data = self.conv(data)                                              # Dim: (n_batch, n_channels * n_bins, n_frames)
        print(data.shape)
        data = data.reshape(data.size(0), self.channels, -1, data.size(-1)) # Dim: (n_batch, n_channels, n_bins, n_frames)
        print(data.shape)
        data = torch.stack([data, stft])                                    # Dim: (n_batch, 2, n_channels, n_bins, n_frames)
        print(data.shape)
        data = data.reshape(data.size(0), -1, data.size(-1))                # Dim: (n_batch, 2 * n_channels * n_bins, n_frames)
        print(data.shape)
        data = self.deconv(data)                                            # Dim: (n_batch, n_channels, out_length)
        print(data.shape)
        data = data[..., :timesteps]                                        # Dim: (n_batch, n_channels, timesteps)
        print(data.shape)
        return data
