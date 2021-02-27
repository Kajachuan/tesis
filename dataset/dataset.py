import musdb
import random
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple

class MUSDB18Dataset(Dataset):
    """
    Dataset MUSDB18 (basado en la implementación en OpenUnmix)
    """
    def __init__(self, base_path: str, subset: str, split: str, target: str,
                 duration: Optional[float], samples: int = 1, random: bool = False,
                 n_fft: int = 4096, hop: int = 1024) -> None:
        """
        base_path -- Ruta del dataset
        subset -- Nombre del conjunto: 'train' o 'test'
        split -- División del entrenamiento: 'train' o 'valid' cuando subset='train'
        target -- Instrumento que se va a separar 'vocals', 'drums', 'bass' o 'vocal'
        duration -- Duración de cada canción en segundos
        samples -- Cantidad de muestras de cada cancion
        random -- True si se van a mezclar las canciones de forma aleatoria
        n_fft -- Tamaño de la fft para el espectrograma
        hop -- Tamaño del hop del espectrograma
        """
        super(MUSDB18Dataset, self).__init__()
        self.sample_rate = 44100 # Por ahora no se usa
        self.split = split
        self.target = target
        self.duration = duration
        self.samples = samples
        self.random = random
        self.n_fft = n_fft
        self.hop = hop
        self.mus = musdb.DB(root=base_path, subsets=subset, split=split)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        track = self.mus[index // self.samples]

        if self.split == 'train' and self.duration:
            track.chunk_duration = self.duration
            track.chunk_start = random.uniform(0, track.duration - self.duration)
        x_audio = track.audio.T
        y_audio = track.targets[self.target].audio.T
        x = torch.as_tensor([np.abs(librosa.stft(x_audio[0], n_fft=self.n_fft, hop_length=self.hop)),
                             np.abs(librosa.stft(x_audio[1], n_fft=self.n_fft, hop_length=self.hop))])
        y = torch.as_tensor([np.abs(librosa.stft(y_audio[0], n_fft=self.n_fft, hop_length=self.hop)),
                             np.abs(librosa.stft(y_audio[1], n_fft=self.n_fft, hop_length=self.hop))])
        return x, y

    def __len__(self) -> int:
        return len(self.mus) * self.samples
