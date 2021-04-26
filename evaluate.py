import argparse
import musdb
import museval
import torch
import tqdm
from separator import *

def main():
    # Solo musdb por ahora !!

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, help="Ruta del modelo a evaluar")
    parser.add_argument("--init", type=int, default=0, choices=range(50), help="Índice de la canción de inicio")
    parser.add_argument("--model", type=str, choices=["spectrogram", "wave"], help="Modelo a utilizar")
    parser.add_argument("--other", action="store_true", help="Utilizar el modelo de other")
    parser.add_argument("--output", type=str, help="Ruta donde se guarda la evaluación")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    parser.add_argument("--vocals", action="store_true", help="Restar vocals para calcular el acompañamiento")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print("GPU disponible:", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if args.model == "spectrogram":
        separator = SpectrogramSeparator(self.checkpoints, self.other, self.vocals)
    else: # wave
        raise NotImplementedError

    print("Cargando canciones de test")
    mus = musdb.DB(root=args.root, subsets='test')

    for i in tqdm.tqdm(range(args.init, 50)):
        track = mus.tracks[i]
        print(f"Canción {i}: {track.name}")
        signal = torch.as_tensor(track.audio.T, dtype=torch.float32).to(device)
        result = separator.separate(signal)
        museval.eval_mus_track(track, result, f"{args.output}")

if __name__ == '__main__':
    main()
