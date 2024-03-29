import argparse
import os
import torch
import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from dataset.dataset import MUSDB18Dataset
from blend_net.blend import BlendNet
from wave_model.model import WaveModel
from attention_model.model import AttentionModel
from spectrogram_model.model import SpectrogramModel

def train(network, train_loader, device, attn_model, stft_model, wave_model, optimizer):
    batch_loss, count = 0, 0
    network.train()
    pbar = tqdm.tqdm(train_loader)
    for x, y in pbar:
        pbar.set_description("Entrenando batch")
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.no_grad():
            args = []
            if attn_model is not None:
                wave_attn, _ = attn_model(x)
                args.append(wave_attn)
            if wave_model is not None:
                wave = wave_model(x)
                args.append(wave)
            if stft_model is not None:
                _, _, wave_stft = stft_model(x)
                args.append(wave_stft)

        y_hat = network(*args)
        loss = mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item() * y.size(0)
        count += y.size(0)
    return batch_loss / count

def valid(network, valid_loader, device, attn_model, stft_model, wave_model):
    batch_loss, count = 0, 0
    network.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(valid_loader)
        for x, y in pbar:
            pbar.set_description("Validando")
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            args = []
            if attn_model is not None:
                wave_attn, _ = attn_model(x)
                args.append(wave_attn)
            if wave_model is not None:
                wave = wave_model(x)
                args.append(wave)
            if stft_model is not None:
                _, _, wave_stft = stft_model(x)
                args.append(wave_stft)

            y_hat = network(*args)
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
    parser.add_argument("--layers", type=int, default=5, help="Número de capas")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("--output", type=str, help="Directorio de salida")
    parser.add_argument("--partitions", type=int, default=1, help="Número de partes de las canciones de validación")
    parser.add_argument("--path-attn", type=str, help="Ruta del modelo de atención")
    parser.add_argument("--path-stft", type=str, help="Ruta del modelo de STFT")
    parser.add_argument("--path-wave", type=str, help="Ruta del modelo de Wave")
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
    blend = 0
    
    if args.path_attn is None:
        attn_model = None
    else:
        print("Cargando modelo de atención")
        state = torch.load(f"{args.path_attn}/{args.target}/best_checkpoint", map_location=device)
        attn_model = AttentionModel(*state["args"]).to(device)
        attn_model.load_state_dict(state["state_dict"])
        attn_model.eval()
        for param in attn_model.parameters():
            param.requires_grad = False
        blend += 1

    if args.path_wave is None:
        wave_model = None
    else:
        print("Cargando modelo de Wave")
        state = torch.load(f"{args.path_wave}/{args.target}/best_checkpoint", map_location=device)
        wave_model = WaveModel(*state["args"]).to(device)
        wave_model.load_state_dict(state["state_dict"])
        wave_model.eval()
        for param in wave_model.parameters():
            param.requires_grad = False
        blend += 1

    if args.path_stft is None:
        stft_model = None
    else:
        print("Cargando modelo de STFT")
        state = torch.load(f"{args.path_stft}/{args.target}/best_checkpoint", map_location=device)
        stft_model = SpectrogramModel(*state["args"]).to(device)
        stft_model.load_state_dict(state["state_dict"])
        stft_model.eval()
        for param in stft_model.parameters():
            param.requires_grad = False
        blend += 1

    nfft = 4096
    hop = 1024
    model_args = [blend, args.layers, args.channels, nfft, hop]
    network = BlendNet(*model_args).to(device)

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
        train_loss = train(network, train_loader, device, attn_model, stft_model, wave_model, optimizer)
        valid_loss = valid(network, valid_loader, device, attn_model, stft_model, wave_model)
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
