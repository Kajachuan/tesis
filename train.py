import argparse
import os
import torch
import tqdm
import torch_xla
import torch_xla.core.xla_model as xm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from dataset.dataset import MUSDB18Dataset
from spectrogram_model.model import SpectrogramModel
from spectrogram_model.stft import STFT
from wave_model.model import WaveModel

def train(network, optimizer, train_loader, device, stft):
    batch_loss, count = 0, 0
    network.train()
    pbar = tqdm.tqdm(train_loader)
    for x, y in pbar:
        pbar.set_description("Entrenando batch")
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()

        if stft is None:
            y_hat = network(x)
        else:
            _, y_hat, _ = network(x)
            y = stft(y)[..., 0]

        loss = mse_loss(y_hat, y)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        batch_loss += loss.item() * y.size(0)
        count += y.size(0)
    return batch_loss / count

def valid(network, valid_loader, device, stft):
    batch_loss, count = 0, 0
    network.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(valid_loader)
        for x, y in pbar:
            pbar.set_description("Validando")
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if stft is None:
                y_hat = network(x)
            else:
                _, y_hat, _ = network(x)
                y = stft(y)[..., 0]

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
    parser.add_argument("--duration", type=float, default=5.0, help="Duración de cada canción")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("--output", type=str, help="Directorio de salida")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    parser.add_argument("--samples", type=int, default=1, help="Muestras por cancion")
    parser.add_argument("--target", type=str, default="vocals", help="Instrumento a separar")
    parser.add_argument("--weight-decay", type=float, default=0, help="Decaimiento de los pesos de Adam")
    parser.add_argument("--workers", type=int, default=0, help="Número de workers para cargar los datos")

    subparsers = parser.add_subparsers(help="Tipo de modelo", dest="model")

    # Modelo de espectrograma
    parser_spec = subparsers.add_parser("spectrogram", help="Modelo de espectrograma")
    parser_spec.add_argument("--dropout", type=float, default=0.3, help="Dropout del BLSTM")
    parser_spec.add_argument("--hidden-size", type=int, default=20, help="Cantidad de unidades BLSTM")
    parser_spec.add_argument("--hop", type=int, default=1024, help="Tamaño del hop del STFT")
    parser_spec.add_argument("--layers", type=int, default=2, help="Cantidad de capas BLSTM")
    parser_spec.add_argument("--nfft", type=int, default=4096, help="Tamaño de la FFT del STFT")

    # Modelo de wave
    parser_wave = subparsers.add_parser("wave", help="Modelo de wave")
    parser_wave.add_argument("--down", type=int, default=1, help="Tamaño del filtro del bloque de downsampling")
    parser_wave.add_argument("--layers", type=int, default=5, help="Cantidad de capas de U-Net")
    parser_wave.add_argument("--filters", type=int, default=10, help="Cantidad de filtros por capa U-Net")
    parser_wave.add_argument("--up", type=int, default=1, help="Tamaño del filtro del bloque de upsampling")

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    device = xm.xla_device()

    if args.model == "spectrogram":
        model_args = [args.channels, args.hidden_size, args.layers, args.dropout, args.nfft, args.hop]
        network = SpectrogramModel(*model_args).to(device)
        stft = STFT(args.nfft, args.hop).to(device)
    elif args.model == "wave":
        model_args = [args.channels, args.layers, args.filters, args.down, args.up]
        network = WaveModel(*model_args).to(device)
        stft = None
    else:
        raise NotImplementedError

    if args.dataset == "musdb":
        train_dataset = MUSDB18Dataset(base_path=args.root, subset="train", split="train", target=args.target,
                                       duration=args.duration, samples=args.samples, random=True)
        valid_dataset = MUSDB18Dataset(base_path=args.root, subset="train", split="valid", target=args.target,
                                       duration=None, samples=1, random=False)
    # elif args.dataset == "medleydb":
    #     pass
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=args.workers, pin_memory=True)

    optimizer = Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    if args.checkpoint:
        state = torch.load(f"{args.checkpoint}/{args.target}/last_checkpoint")
        xm.send_cpu_data_to_device(state, xm.xla_device())
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
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    t = tqdm.trange(initial_epoch, args.epochs + 1)
    for epoch in t:
        t.set_description("Entrenando iteración")
        train_loss = train(network, optimizer, train_loader, device, stft)
        valid_loss = valid(network, valid_loader, device, stft)
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
