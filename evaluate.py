import argparse
import json
import musdb
import museval
import os
import torch
import tqdm
from math import ceil
from separator import *

def merge_jsons(out_dir: str, track_name: str, partitions: int) -> None:
    data = []
    for i in range(1, partitions + 1):
        f = open(f"{out_dir}{i}/test/{track_name}.json")
        data.append(json.load(f))
        f.close()

    for i in range(1, partitions):
        for target1 in data[0]["targets"]:
            for target2 in data[i]["targets"]:
                if target2["name"] == target1["name"]:
                    frames1 = target1["frames"]
                    frames2 = target2["frames"]
                    for frame in frames2:
                        frame["time"] += len(frames1)
                    target1["frames"] = frames1 + frames2
                    break

    with open(f"{out_dir}/test/{track_name}.json", "w") as out:
        json.dump(data[0], out, indent=2)

    for i in range(1, partitions + 1):
        os.remove(f"{out_dir}{i}/test/{track_name}.json")

def main():
    # Solo musdb por ahora !!

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, help="Ruta del modelo a evaluar")
    parser.add_argument("--end", type=int, default=49, choices=range(50), help="Índice de la canción de fin")
    parser.add_argument("--init", type=int, default=0, choices=range(50), help="Índice de la canción de inicio")
    parser.add_argument("--other", action="store_true", help="Utilizar el modelo de other")
    parser.add_argument("--output", type=str, help="Ruta donde se guarda la evaluación")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    parser.add_argument("--seconds", type=int, default=60, help="Segundos de cada partición")
    parser.add_argument("--vocals", action="store_true", help="Restar vocals para calcular el acompañamiento")

    subparsers = parser.add_subparsers(help="Tipo de modelo", dest="model")
    parser_spec = subparsers.add_parser("spectrogram", help="Modelo de espectrograma")
    parser_wave = subparsers.add_parser("wave", help="Modelo de wave")
    parser_attn = subparsers.add_parser("attention", help="Modelo de atención")

    parser_blend = subparsers.add_parser("blend", help="Modelo de mezcla")
    parser_blend.add_argument("--checkpoints-attn", type=str, help="Ruta del modelo de atención")
    parser_blend.add_argument("--checkpoints-wave", type=str, help="Ruta del modelo de wave")
    parser_blend.add_argument("--checkpoints-stft", type=str, help="Ruta del modelo de espectrograma")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print("GPU disponible:", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if args.model == "spectrogram":
        separator = SpectrogramSeparator(args.checkpoints, args.other, args.vocals, device)
    elif args.model == "wave":
        separator = WaveSeparator(args.checkpoints, args.other, args.vocals, device)
    elif args.model == "attention":
        separator = AttentionSeparator(args.checkpoints, args.other, args.vocals, device)
    elif args.model == "blend":
        separator = BlendSeparator(args.checkpoints_attn, args.checkpoints_wave, 
                                   args.checkpoints_stft, root=args.checkpoints, 
                                   use_other=args.other, use_vocals=args.vocals, 
                                   device=device)
    else:
        raise NotImplementedError

    print("Cargando canciones de test")
    mus = musdb.DB(root=args.root, subsets='test')

    os.makedirs(f"{args.output}/test", exist_ok=True)

    for i in tqdm.tqdm(range(args.init, args.end + 1)):
        track = mus.tracks[i]
        print(f"Canción {i}: {track.name}")

        partitions = ceil(track.duration / args.seconds)
        print(f"{partitions} particiones")

        for i in range(1, partitions):
            print(f"Partición {i}: {args.seconds} segundos")
            track.chunk_start = ((i - 1) % partitions) * args.seconds
            track.chunk_duration = args.seconds
            signal = torch.as_tensor(track.audio.T, dtype=torch.float32).to(device)
            result = separator.separate(signal)
            museval.eval_mus_track(track, result, f"{args.output}{i}")

        track.chunk_start = (partitions - 1) * args.seconds
        track.chunk_duration = track.duration - track.chunk_start
        print(f"Partición {partitions}: {track.chunk_duration} segundos")
        signal = torch.as_tensor(track.audio.T, dtype=torch.float32).to(device)
        result = separator.separate(signal)
        museval.eval_mus_track(track, result, f"{args.output}{partitions}")

        merge_jsons(args.output, track.name, partitions)

        for i in range(1, partitions + 1):
            os.rmdir(f"{args.output}{i}/test")
            os.rmdir(f"{args.output}{i}")

if __name__ == '__main__':
    main()
