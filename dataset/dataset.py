import musdb
import random
import torch
import numpy as np
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
            trach.chunk_start = (index % self.partitions) * chunk
            if (index + 1) % self.partitions == 0:
                track.chunk_duration = track.duration - track.chunk_start
            else:
                track.chunk_duration = chunk

            x = torch.as_tensor(track.audio.T, dtype=torch.float32)
            y = torch.as_tensor(track.targets[self.target].audio.T, dtype=torch.float32)
        return x, y

    def __len__(self) -> int:
        return len(self.mus) * self.samples * self.partitions
