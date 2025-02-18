import argparse
import os
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.optim as optim
import wandb
from cv2 import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AnimeDataSet
from modeling.anime_gan import AnimeGanDiscriminator, AnimeGanGenerator
from modeling.losses import AnimeGanLoss, LossSummary
from utils.common import load_checkpoint, save_checkpoint, set_lr
from utils.image_processing import denormalize_input

gaussian_mean = torch.tensor(0.0)
gaussian_std = torch.tensor(0.1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="style")
    parser.add_argument("--photo_dir", type=str)
    parser.add_argument("--style_dir", type=str)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--data-dir", type=str, default="/content/dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--init-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--checkpoint-dir", type=str, default="/content/checkpoints")
    parser.add_argument("--save-image-dir", type=str, default="/content/images")
    parser.add_argument(
        "--gan-loss", type=str, default="lsgan", help="lsgan / hinge / bce"
    )
    parser.add_argument("--resume", type=str, default="False")
    parser.add_argument("--use_sn", action="store_true")
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--debug-samples", type=int, default=0)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=4e-4)
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument(
        "--wadvg", type=float, default=10.0, help="Adversarial loss weight for G"
    )
    parser.add_argument(
        "--wadvd", type=float, default=10.0, help="Adversarial loss weight for D"
    )
    parser.add_argument("--wcon", type=float, default=1.5, help="Content loss weight")
    parser.add_argument("--wgra", type=float, default=3.0, help="Gram loss weight")
    parser.add_argument("--wcol", type=float, default=30.0, help="Color loss weight")
    parser.add_argument(
        "--d-layers", type=int, default=3, help="Discriminator conv layers"
    )
    parser.add_argument("--d-noise", action="store_true")

    return parser.parse_args()


def collate_fn(batch):
    # img, anime, anime_gray, anime_smt_gray = zip(*batch)
    # return (
    #     torch.stack(img, 0),
    #     torch.stack(anime, 0),
    #     torch.stack(anime_gray, 0),
    #     torch.stack(anime_smt_gray, 0),
    # )

    img, anime, anime_gray = zip(*batch)
    return (
        torch.stack(img, 0),
        torch.stack(anime, 0),
        torch.stack(anime_gray, 0),
        # torch.stack(anime_smt_gray, 0),
    )


def check_params(args):
    # data_path = os.path.join(args.data_dir, args.dataset)
    # if not os.path.exists(data_path):
    #     raise FileNotFoundError(f"Dataset not found {data_path}")

    if not os.path.exists(args.save_image_dir):
        print(f"* {args.save_image_dir} does not exist, creating...")
        os.makedirs(args.save_image_dir)

    if not os.path.exists(args.checkpoint_dir):
        print(f"* {args.checkpoint_dir} does not exist, creating...")
        os.makedirs(args.checkpoint_dir)

    assert args.gan_loss in {
        "lsgan",
        "hinge",
        "bce",
    }, f"{args.gan_loss} is not supported"


def save_samples(
    generator, loader, args, max_imgs=2, subname="gen", device=torch.device("cpu")
):
    """
    Generate and save images
    """
    generator.eval()

    max_iter = (max_imgs // args.batch_size) + 1
    fake_imgs = []
    real_imgs = []

    for i, (img, *_) in enumerate(loader):
        with torch.no_grad():
            fake_img = generator(img.to(device))
            fake_img = fake_img.detach().cpu().numpy()
            # Channel first -> channel last
            fake_img = fake_img.transpose(0, 2, 3, 1)
            fake_imgs.append(denormalize_input(fake_img, dtype=np.int16))

            real_img = img.detach().cpu().numpy().transpose(0, 2, 3, 1)
            real_imgs.append(denormalize_input(real_img, dtype=np.int16))

        if i + 1 == max_iter:
            break

    fake_imgs = np.concatenate(fake_imgs, axis=0)
    real_imgs = np.concatenate(real_imgs, axis=0)

    for i, curr_fake_img in enumerate(fake_imgs):
        save_path = os.path.join(args.save_image_dir, f"{subname}_{i}.jpg")

        curr_real_image_wandb = wandb.Image(real_imgs[i], caption="Real image")
        curr_fake_image_wandb = wandb.Image(curr_fake_img, caption="Generated image")
        wandb.log(
            {
                "real_img": curr_real_image_wandb,
                "fake_img": curr_fake_image_wandb,
            }
        )

        cv2.imwrite(save_path, curr_fake_img[..., ::-1])


def gaussian_noise():
    return torch.normal(gaussian_mean, gaussian_std)


def main(args):
    check_params(args)
    wandb.config.update(args)

    print("Init models...")

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")

    G = AnimeGanGenerator(args.dataset).to(device)
    D = AnimeGanDiscriminator(args).to(device)

    loss_tracker = LossSummary()

    loss_fn = AnimeGanLoss(args, device=device)

    # Create DataLoader
    data_loader = DataLoader(
        AnimeDataSet(args),
        batch_size=args.batch_size,
        num_workers=cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    start_e = 0
    if args.resume == "GD":
        # Load G and D
        try:
            start_e = load_checkpoint(G, args.checkpoint_dir)
            print("G weight loaded")
            load_checkpoint(D, args.checkpoint_dir)
            print("D weight loaded")
        except Exception as e:
            print("Could not load checkpoint, train from scratch", e)
    elif args.resume == "G":
        # Load G only
        try:
            start_e = load_checkpoint(G, args.checkpoint_dir, posfix="_init")
        except Exception as e:
            print("Could not load G init checkpoint, train from scratch", e)

    for epoch_num in range(start_e, args.epochs):
        print(f"Epoch {epoch_num}/{args.epochs}")
        bar = tqdm(data_loader)
        G.train()

        init_losses = []

        if epoch_num < args.init_epochs:
            # Train with content loss only
            set_lr(optimizer_g, args.init_lr)
            for img, *_ in bar:
                img = img.to(device)

                optimizer_g.zero_grad()

                fake_img = G(img)
                loss = loss_fn.content_loss_vgg(img, fake_img)
                loss.backward()
                optimizer_g.step()

                init_losses.append(loss.cpu().detach().numpy())
                avg_content_loss = sum(init_losses) / len(init_losses)
                bar.set_description(
                    f"[Init Training G] content loss: {avg_content_loss:2f}"
                )

            set_lr(optimizer_g, args.lr_g)
            save_checkpoint(G, optimizer_g, epoch_num, args, posfix="_init")
            save_samples(G, data_loader, args, subname="initg", device=device)
            continue

        loss_tracker.reset()
        # for img, anime, anime_gray, anime_smt_gray in bar:
        for img, anime, anime_gray in bar:
            # To cuda
            img = img.to(device)
            anime = anime.to(device)
            anime_gray = anime_gray.to(device)
            # anime_smt_gray = anime_smt_gray.to(device)

            # ---------------- TRAIN D ---------------- #
            optimizer_d.zero_grad()
            fake_img = G(img).detach()

            # Add some Gaussian noise to images before feeding to D
            if args.d_noise:
                fake_img += gaussian_noise()
                anime += gaussian_noise()
                anime_gray += gaussian_noise()
                # anime_smt_gray += gaussian_noise()

            fake_d = D(fake_img)
            real_anime_d = D(anime)
            real_anime_gray_d = D(anime_gray)
            # real_anime_smt_gray_d = D(anime_smt_gray)

            loss_d = loss_fn.compute_loss_D(
                fake_d,
                real_anime_d,
                real_anime_gray_d,
                # real_anime_smt_gray_d
            )

            wandb.log({"loss_d": loss_d})

            loss_d.backward()
            optimizer_d.step()

            loss_tracker.update_loss_D(loss_d)

            # ---------------- TRAIN G ---------------- #
            optimizer_g.zero_grad()

            fake_img = G(img)
            fake_d = D(fake_img)

            adv_loss, con_loss, gra_loss, col_loss = loss_fn.compute_loss_G(
                fake_img, img, fake_d, anime_gray
            )

            loss_g = adv_loss + con_loss + gra_loss + col_loss

            wandb.log({"loss_g": loss_g})

            loss_g.backward()
            optimizer_g.step()

            loss_tracker.update_loss_G(adv_loss, gra_loss, col_loss, con_loss)

            avg_adv, avg_gram, avg_color, avg_content = loss_tracker.avg_loss_G()
            avg_adv_d = loss_tracker.avg_loss_D()
            bar.set_description(
                f"loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} / loss D: {avg_adv_d:2f}"
            )

        if epoch_num % args.save_interval == 0:
            save_checkpoint(G, optimizer_g, epoch_num, args)
            save_checkpoint(D, optimizer_d, epoch_num, args)
            save_samples(G, data_loader, args, device=device)


if __name__ == "__main__":
    wandb.init(project="anime-gan", entity="vadbeg")
    args = parse_args()

    print("# ==== Train Config ==== #")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("==========================")

    main(args)
