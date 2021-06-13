import musdb
import os
import random
import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
from typing import Optional, Tuple

class MUSDB18Dataset(Dataset):
    """
    Dataset MUSDB18 (basado en la implementación en OpenUnmix)
    """
    def __init__(self, base_path: str, subset: str, split: str, target: str, duration: Optional[float],
                 samples: int = 1, random: bool = False, partitions: int = 1) -> None:
        """
        base_path -- Ruta del dataset
        subset -- Nombre del conjunto: 'train' o 'test'
        split -- División del entrenamiento: 'train' o 'valid' cuando subset='train'
        target -- Instrumento que se va a separar 'vocals', 'drums', 'bass' o 'vocal'
        duration -- Duración de cada canción en segundos
        samples -- Cantidad de muestras de cada cancion
        random -- True si se van a mezclar las canciones de forma aleatoria
        partitions -- Cantidad de particiones de las canciones de validación
        """
        super(MUSDB18Dataset, self).__init__()
        self.sample_rate = 44100 # Por ahora no se usa
        self.split = split
        self.target = target
        self.duration = duration
        self.samples = samples
        self.random = random
        self.partitions = partitions
        self.mus = musdb.DB(root=base_path, subsets=subset, split=split)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split == 'train' and self.duration:
            sources = []
            target_idx = None
            track = self.mus[index // self.samples]

            for idx, source in enumerate(self.mus.setup['sources']):
                if source == self.target:
                    target_idx = idx

                if self.random:
                    track = random.choice(self.mus)

                track.chunk_duration = self.duration
                track.chunk_start = random.uniform(0, track.duration - self.duration)

                audio = track.sources[source].audio.T
                audio *= np.random.uniform(0.25, 1.25, (audio.shape[0], 1))
                if random.random() < 0.5:
                    audio = np.flipud(audio)
                audio = torch.as_tensor(audio.copy(), dtype=torch.float32)
                sources.append(audio)

            stems = torch.stack(sources, dim=0)
            x = stems.sum(0)
            y = stems[target_idx]

        # Validación y Test
        else:
            track = self.mus[index // self.partitions]

            chunk = track.duration // self.partitions
            track.chunk_start = (index % self.partitions) * chunk
            if (index + 1) % self.partitions == 0:
                track.chunk_duration = track.duration - track.chunk_start
            else:
                track.chunk_duration = chunk

            x = torch.as_tensor(track.audio.T, dtype=torch.float32)
            y = torch.as_tensor(track.targets[self.target].audio.T, dtype=torch.float32)
        return x, y

    def __len__(self) -> int:
        return len(self.mus) * self.samples * self.partitions

class MedleyDBDataset(Dataset):
    """
    Dataset MedleyDB

    La estructura del directorio debe ser:

    - MedleyDB
        - mixes
            - (wav)
            ...
        - stems
            - drums
                - (wav)
                ...
            - vocals
                - (wav)
                ...
            ...
    """
    def __init__(self, base_path: str, split: str, target: str, duration: Optional[float],
                 samples: int = 1, partitions: int = 1) -> None:
        """
        base_path -- Ruta del dataset
        split -- División del entrenamiento: 'train' o 'valid' cuando subset='train'
        target -- Instrumento que se va a separar 'vocals', 'drums', 'bass' o 'vocal'
        duration -- Duración de cada canción en segundos
        samples -- Cantidad de muestras de cada cancion
        partitions -- Cantidad de particiones de las canciones de validación
        """
        super(MedleyDBDataset, self).__init__()
        self.sample_rate = 44100
        self.base_path = base_path
        self.split = split
        self.duration = duration * self.sample_rate if duration else None
        self.samples = samples
        self.partitions = partitions
        self.track_names = os.listdir(f'{base_path}/stems/{target}/{split}')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == 'train':
            track_name = self.track_names[index // self.samples]
            _, mix = wavfile.read(f'{self.base_path}/mixes/{self.target}/{track_name}')
            _, source = wavfile.read(f'{self.base_path}/stems/{self.target}/{split}/{track_name}')

            start = random.uniform(0, mix.size(0) - self.duration)
            mix = mix[start:start + self.duration, :].T
            source = source[start:start + self.duration, :].T

            vol = np.random.uniform(0.25, 1.25, (mix.size(0), 1))
            mix *= vol
            source *= vol

            if random.random() < 0.5:
                mix = np.flipud(mix)
                source = np.flipud(source)

            x = torch.as_tensor(mix, dtype=torch.float32)
            y = torch.as_tensor(source, dtype=torch.float32)
        else:
            track_name = self.track_names[index // self.partitions]
            _, mix = wavfile.read(f'{self.base_path}/mixes/{self.target}/{track_name}')
            _, source = wavfile.read(f'{self.base_path}/stems/{self.target}/{split}/{track_name}')

            chunk = mix.size(0) // self.partitions
            chunk_start = (index % self.partitions) * chunk
            if (index + 1) % self.partitions == 0:
                chunk_duration = mix.size(0) - chunk_start
            else:
                chunk_duration = chunk

            mix = mix[chunk_start:chunk_start + chunk_duration, :].T
            source = source[chunk_start:chunk_start + chunk_duration, :].T

            x = torch.as_tensor(mix, dtype=torch.float32)
            y = torch.as_tensor(source, dtype=torch.float32)
        return x, y

    def __len__(self) -> int:
        return len(self.track_names) * self.samples * self.partitions
