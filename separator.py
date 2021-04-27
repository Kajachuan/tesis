import torch
from torch.nn import Module
from typing import Dict
from numpy import ndarray
from spectrogram_model.model import SpectrogramModel
from wave_model.model import WaveModel

class SpectrogramSeparator:
    def __init__(self, root: str, use_other: bool, use_vocals: bool) -> None:
        """
        Argumentos:
            root -- Directorio donde se encuentran los checkpoints
            use_other -- True si se utiliza el modelo entrenado para "other"
                         False si se calcula "other" como la resta con "vocals" + "drums" + "bass"
            use_vocals -- True si se calcula "accompaniment" como la resta con "vocals"
                          False si se calcula "accompaniment" como "drums" + "bass" + "other"
        """
        self.root = root
        self.use_other = use_other
        self.use_vocals = use_vocals

        self.stems = ["vocals", "drums", "bass"]
        if self.use_other:
            self.stems.append("other")

        self.models = {}
        for stem in self.stems:
            self.load_model(stem)

    def separate(self, track: torch.Tensor) -> Dict[str, ndarray]:
        result = {}
        for stem in self.stems:
            _, _, estim = self.models[stem](track.unsqueeze(0))
            result[stem] = estim[0, ...].cpu().detach().numpy().T
        track = track.cpu().detach().numpy().T

        if not self.use_other:
            result["other"] = track - (result["vocals"] + result["drums"] + result["bass"])

        if self.use_vocals:
            result["accompaniment"] = track - result["vocals"]
        else:
            result["accompaniment"] = result["drums"] + result["bass"] + result["other"]

        return result

    def load_model(self, stem: str) -> None:
        print(f"Cargando modelo de {stem}")
        state = torch.load(f"{self.root}/{stem}/best_checkpoint")
        self.models[stem] = SpectrogramModel(*state["args"]).to(device)
        self.models[stem].load_state_dict(state["state_dict"])
        self.models[stem].eval()
