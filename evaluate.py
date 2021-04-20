import argparse
import musdb
import museval
import torch
import tqdm
from model.model import Model

def main():
    # Solo musdb por ahora !!

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, help="Ruta del modelo a evaluar")
    parser.add_argument("--init", type=int, default=0, choices=range(50), help="Índice de la canción de inicio")
    parser.add_argument("--output", type=str, help="Ruta donde se guarda la evaluación")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print("GPU disponible:", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    stems = ["vocals", "drums", "bass", "other"]
    models = {}
    print("Cargando modelos")
    for stem in stems:
        state = torch.load(f"{args.checkpoints}/{stem}/best_checkpoint")
        models[stem] = Model(*state["args"]).to(device)
        models[stem].load_state_dict(state["state_dict"])
        models[stem].eval()

    print("Cargando canciones de test")
    mus = musdb.DB(root=args.root, subsets='test')

    for i in tqdm.tqdm(range(args.init, 50)):
        track = mus.tracks[i]
        print(f"Canción {i}: {track.name}")
        signal = torch.as_tensor(track.audio.T, dtype=torch.float32).to(device)
        result = {}
        for stem in stems:
            result[stem] = models[stem](signal.unsqueeze(0)).cpu().numpy()[0, ...].T
        museval.eval_mus_track(track, result, f"{args.output}/{track.name}")

if __name__ == '__main__':
    main()
