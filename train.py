import argparse
import json
import torch
import torchaudio
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.dataset import MUSDB18Dataset
from model.model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10, help="Tamaño del batch")
    parser.add_argument("--channels", type=int, default=2, help="Número de canales de audio")
    parser.add_argument("--checkpoint", type=str, help="Directorio de los checkpoints")
    parser.add_argument("--dataset", type=str, default="musdb", choices=["musdb", "medleydb"], help="Nombre del dataset")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout del BLSTM")
    parser.add_argument("--duration", type=float, default=5.0, help="Duración de cada canción")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--hidden-size", type=int, default=20, help="Cantidad de unidades BLSTM")
    parser.add_argument("--hop", type=int, default=1024, help="Tamaño del hop del STFT")
    parser.add_argument("--layers", type=int, default=2, help="Cantidad de capas BLSTM")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("--nfft", type=int, default=4096, help="Tamaño de la FFT del STFT")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    parser.add_argument("--samples", type=int, default=1, help="Muestras por cancion")
    parser.add_argument("--target", type=str, default="vocals", help="Instrumento a separar")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print("GPU disponible:", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if args.dataset == "musdb":
        train_dataset = MUSDB18Dataset(base_path=args.root, subset="train", split="train", target=args.target,
                                       duration=args.duration, samples=args.samples, random=True)
        valid_dataset = MUSDB18Dataset(base_path=args.root, subset="train", split="valid", target=args.target,
                                       duration=None, samples=1, random=True)
    # elif args.dataset == "medleydb":
    #     pass
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    network = Model(n_channels=args.channels, hidden_size=args.hidden_size, num_layers=args.layers,
                    dropout=args.dropout, n_fft=args.nfft, hop=args.hop).to(device)
    optimizer = Adam(network.parameters(), lr=args.learning_rate)

    if args.checkpoint:
        state = torch.load(f"{args.checkpoint}/last_checkpoint", map_location=device)
        network.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])

        train_losses = state["train_losses"]
        valid_losses = state["valid_losses"]
        initial_epoch = state["epoch"] + 1

    else:
        train_losses = []
        valid_losses = []
        initial_epoch = 1

    t = tqdm.trange(initial_epoch, args.epochs + 1)
    for epoch in t:
        t.set_description(f"Época {epoch}")
        train_loss = train() # Implementar
        valid_loss = valid() # Implementar
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

        state = {
            "epoch": epoch,
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_losses": train_losses,
            "valid_losses", valid_losses
        }

        # Falta guardar el mejor
        torch.save(state, f"{args.checkpoint}/last_checkpoint")

if __name__ == '__main__':
    main()
