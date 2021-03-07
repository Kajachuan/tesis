import argparse
import musdb
import museval
import torch
import tqdm
from model.model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Checkpoint del modelo a evaluar")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    parser.add_argument("--target", type=str, help="Instrumento a separar")
    args = parser.parse_args()

    mus = musdb.DB(root=args.root, subsets='test')

    use_cuda = torch.cuda.is_available()
    print("GPU disponible:", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    state = torch.load(args.checkpoint)
    network = Model(*state["args"]).to(device)
    network.load_state_dict(state["state_dict"])
    network.eval()

    results = museval.EvalStore()
    with torch.no_grad():
        for track in tqdm.tqdm(mus.tracks):
            signal = torch.as_tensor(track.audio.T, dtype=torch.float32).to(device)
            _, estimate = network(signal.unsqueeze(0))
            vocals = estimate.cpu().numpy()[0, ...].T
            estimates = {args.target: vocals, 'accompaniment': track.audio - vocals}
            score = museval.eval_mus_track(track, estimates)
            results.add_track(score)

    print(results)

if __name__ == '__main__':
    main()
