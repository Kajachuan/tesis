import torch
from torch.nn import Module
from typing import Dict, List
from numpy import ndarray
from spectrogram_model.model import SpectrogramModel
from wave_model.model import WaveModel

class Separator:
    def __init__(self, root: str, use_other: bool, use_vocals: bool, device: torch.device) -> None:
        """
        Argumentos:
            root -- Directorio donde se encuentran los checkpoints
            use_other -- True si se utiliza el modelo entrenado para "other"
                         False si se calcula "other" como la resta con "vocals" + "drums" + "bass"
            use_vocals -- True si se calcula "accompaniment" como la resta con "vocals"
                          False si se calcula "accompaniment" como "drums" + "bass" + "other"
            device -- Device de PyTorch a utilizar
        """
        self.root = root
        self.use_other = use_other
        self.use_vocals = use_vocals
        self.device = device

        self.stems = ["vocals", "drums", "bass"]
        if self.use_other:
            self.stems.append("other")

        self.models = {}
        for stem in self.stems:
            print(f"Cargando modelo de {stem}")
            state = torch.load(f"{self.root}/{stem}/best_checkpoint")
            self.set_model(stem, state["args"])
            self.models[stem].load_state_dict(state["state_dict"])
            self.models[stem].eval()

    def separate(self, track: torch.Tensor) -> Dict[str, ndarray]:
        result = {}
        for stem in self.stems:
            estim = self.eval_model(stem, track)
            result[stem] = estim.cpu().detach().numpy().T
        track = track.cpu().detach().numpy().T

        if not self.use_other:
            result["other"] = track - (result["vocals"] + result["drums"] + result["bass"])

        if self.use_vocals:
            result["accompaniment"] = track - result["vocals"]
        else:
            result["accompaniment"] = result["drums"] + result["bass"] + result["other"]

        return result

    def set_model(self, stem: str) -> None:
        raise NotImplementedError

    def eval_model(self, stem: str, track: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SpectrogramSeparator(Separator):
    def set_model(self, stem: str, args: List[str]) -> None:
        self.models[stem] = SpectrogramModel(*args).to(self.device)

    def eval_model(self, stem: str, track: torch.Tensor) -> torch.Tensor:
        _, _, estim = self.models[stem](track.unsqueeze(0))
        return estim[0, ...]

class WaveSeparator(Separator):
    def set_model(self, stem: str, args: List[str]) -> None:
        self.models[stem] = WaveModel(*args).to(self.device)

    def eval_model(self, stem: str, track: torch.Tensor) -> torch.Tensor:
        return self.models[stem](track.unsqueeze(0))[0, ...]
