import argparse
import os
import torch
import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from dataset.dataset import MUSDB18Dataset
from attention_model.model import AttentionModel

def train(network, nfft, hop, train_loader, device, optimizer):
    batch_loss, count = 0, 0
    network.train()
    pbar = tqdm.tqdm(train_loader)
    window = torch.hann_window(nfft, device=device)
    for x, y in pbar:
        pbar.set_description("Entrenando batch")
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        _, stft_hat = network(x)
        stft = torch.stft(y.reshape(-1, y.size(-1)), n_fft=nfft, 
                          hop_length=hop, window=window, 
                          onesided=True, return_complex=True)
        stft = torch.stack((torch.real(stft), torch.imag(stft)), dim=-1)
        loss = mse_loss(stft_hat, stft)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item() * y.size(0)
        count += y.size(0)
    return batch_loss / count

def valid(network, nfft, hop, valid_loader, device):
    batch_loss, count = 0, 0
    network.eval()
    window = torch.hann_window(nfft, device=device)
    with torch.no_grad():
        pbar = tqdm.tqdm(valid_loader)
        for x, y in pbar:
            pbar.set_description("Validando")
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _, stft_hat = network(x)
            stft = torch.stft(y.reshape(-1, y.size(-1)), n_fft=nfft, 
                          hop_length=hop, window=window, 
                          onesided=True, return_complex=True)
            stft = torch.stack((torch.real(stft), torch.imag(stft)), dim=-1)
            loss = mse_loss(stft_hat, stft)
            batch_loss += loss.item() * y.size(0)
            count += y.size(0)
        return batch_loss / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10, help="Tamaño del batch")
    parser.add_argument("--checkpoint", type=str, help="Directorio de los checkpoints")
    parser.add_argument("--dataset", type=str, default="musdb", choices=["musdb", "medleydb"], help="Nombre del dataset")
    parser.add_argument("--duration", type=float, default=5.0, help="Duración de cada canción")
    parser.add_argument("--epochs", type=int, default=10, help="Número de iteraciones")
    parser.add_argument("--hop", type=int, default=1024, help="Tamaño del hop del STFT")
    parser.add_argument("--layers", type=int, default=5, help="Número de capas")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("--nfft", type=int, default=4096, help="Tamaño de la FFT del STFT")
    parser.add_argument("--output", type=str, help="Directorio de salida")
    parser.add_argument("--partitions", type=int, default=1, help="Número de partes de las canciones de validación")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    parser.add_argument("--samples", type=int, default=1, help="Muestras por cancion")
    parser.add_argument("--target", type=str, default="vocals", help="Instrumento a separar")
    parser.add_argument("--weight-decay", type=float, default=0, help="Decaimiento de los pesos de Adam")
    parser.add_argument("--workers", type=int, default=0, help="Número de workers para cargar los datos")

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    use_cuda = torch.cuda.is_available()
    print("GPU disponible:", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model_args = [args.nfft, args.hop]
    network = AttentionModel(*model_args).to(device)

    if args.dataset == "musdb":
        train_dataset = MUSDB18Dataset(base_path=args.root, subset="train", split="train", target=args.target,
                                       duration=args.duration, samples=args.samples, random=True)
        valid_dataset = MUSDB18Dataset(base_path=args.root, subset="train", split="valid", target=args.target,
                                       duration=None, samples=1, random=False, partitions=args.partitions)
    # elif args.dataset == "medleydb":
    #     pass
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=args.workers, pin_memory=True)

    optimizer = Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    if args.checkpoint:
        state = torch.load(f"{args.checkpoint}/{args.target}/last_checkpoint", map_location=device)
        network.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        train_losses = state["train_losses"]
        valid_losses = state["valid_losses"]
        initial_epoch = state["epoch"] + 1
        best_loss = state["best_loss"]
    else:
        train_losses = []
        valid_losses = []
        initial_epoch = 1
        best_loss = float("inf")

    out_path = f"{args.output}/{args.target}"
    os.makedirs(out_path, exist_ok=True)

    t = tqdm.trange(initial_epoch, args.epochs + 1)
    for epoch in t:
        t.set_description("Entrenando iteración")
        train_loss = train(network, args.nfft, args.hop, train_loader, device, optimizer)
        valid_loss = valid(network, args.nfft, args.hop, valid_loader, device)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

        state = {
            "args": model_args,
            "epoch": epoch,
            "best_loss": best_loss,
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_losses": train_losses,
            "valid_losses": valid_losses
        }

        if valid_loss < best_loss:
            best_loss = valid_loss
            state["best_loss"] = best_loss
            torch.save(state, f"{out_path}/best_checkpoint")
        torch.save(state, f"{out_path}/last_checkpoint")

if __name__ == '__main__':
    main()
