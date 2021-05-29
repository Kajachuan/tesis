import argparse
import json
import musdb
import museval
import os
import torch
import tqdm
from separator import *

def merge_json(out_dir: str, track_name: str) -> None:
    with open(f"{out_dir}1/test/{track_name}.json") as f1, open(f"{out_dir}2/test/{track_name}.json") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    for target1 in data1["targets"]:
        for target2 in data2["targets"]:
            if target2["name"] == target1["name"]:
                frames1 = target1["frames"]
                frames2 = target2["frames"]
                for frame in frames2:
                    frame["time"] += len(frames1)
                target1["frames"] = frames1 + frames2
                break

    with open(f"{out_dir}/test/{track_name}.json", "w") as out:
        json.dump(data1, out, indent=2)

    os.remove(f"{out_dir}1/{track_name}.json")
    os.remove(f"{out_dir}2/{track_name}.json")

def main():
    # Solo musdb por ahora !!

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, help="Ruta del modelo a evaluar")
    parser.add_argument("--end", type=int, default=49, choices=range(50), help="Índice de la canción de fin")
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
        separator = SpectrogramSeparator(args.checkpoints, args.other, args.vocals, device)
    elif args.model == "wave":
        separator = WaveSeparator(args.checkpoints, args.other, args.vocals, device)
    else:
        raise NotImplementedError

    print("Cargando canciones de test")
    mus = musdb.DB(root=args.root, subsets='test')

    for i in tqdm.tqdm(range(args.init, args.end + 1)):
        track = mus.tracks[i]
        print(f"Canción {i}: {track.name}")

        # Primera mitad
        print("Primera mitad")
        half = track.duration // 2
        track.chunk_duration = half
        signal = torch.as_tensor(track.audio.T, dtype=torch.float32).to(device)
        result = separator.separate(signal)
        museval.eval_mus_track(track, result, f"{args.output}1")

        # Segunda mitad
        print("Segunda mitad")
        track.chunk_start = half
        track.chunk_duration = track.duration - half
        signal = torch.as_tensor(track.audio.T, dtype=torch.float32).to(device)
        result = separator.separate(signal)
        museval.eval_mus_track(track, result, f"{args.output}2")

        merge_json(args.output, track.name)

    os.rmdir(f"{args.output}1")
    os.rmdir(f"{args.output}2")

if __name__ == '__main__':
    main()
