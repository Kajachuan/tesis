import argparse
import json
import torch
import torchaudio
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from dataset.dataset import MUSDB18Dataset
from model.model import Model

def train(network, optimizer, train_loader, device):
    batch_loss = 0
    count = 0
    network.train()
    pbar = tqdm.tqdm(train_loader)
    for x, y in pbar:
        pbar.set_description("Entrenando batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        m, y_hat = network(x)
        loss = mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item() * y.size(0)
        count += y.size(0)
    return batch_loss / count

def valid(network, optimizer, valid_loader, device):
    batch_loss = 0
    count = 0
    network.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            m, y_hat = network(x)
            loss = mse_loss(y_hat, y)
            batch_loss += loss.item() * y.size(0)
            count += y.size(0)
        return batch_loss / count

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
                                       duration=None, samples=1, random=False)
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
        best_loss = state["best_loss"]

    else:
        train_losses = []
        valid_losses = []
        initial_epoch = 1
        best_loss = float("inf")

    t = tqdm.trange(initial_epoch, args.epochs + 1)
    for epoch in t:
        t.set_description(f"Entrenando época {epoch}")
        train_loss = train(network, optimizer, train_loader, device)
        valid_loss = valid(network, optimizer, valid_loader, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

        state = {
            "epoch": epoch,
            "best_loss": best_loss,
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_losses": train_losses,
            "valid_losses": valid_losses
        }

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(state, f"{args.checkpoint}/best_checkpoint")
        torch.save(state, f"{args.checkpoint}/last_checkpoint")

if __name__ == '__main__':
    main()
