import os
import torch
import torchvision.transforms as transforms
import numpy as np
from dataset import ReferenceTargetDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from deep_utils import show_destroy_cv2

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True, desc=f"{config.DATASET_NAME} TRAIN, Epoch {epoch}/{config.NUM_EPOCHS}: ")

    for idx, (target, reference) in enumerate(loop):
        target = target.to(config.DEVICE)
        reference = reference.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_reference = gen_H(target)
            D_H_real = disc_H(reference)
            D_H_fake = disc_H(fake_reference.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_target = gen_Z(reference)
            D_Z_real = disc_Z(target)
            D_Z_fake = disc_Z(fake_target.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_reference)
            D_Z_fake = disc_Z(fake_target)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_target = gen_Z(fake_reference)
            cycle_reference = gen_H(fake_target)
            cycle_target_loss = l1(target, cycle_target)
            cycle_reference_loss = l1(reference, cycle_reference)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_target = gen_Z(target)
            identity_reference = gen_H(reference)
            identity_target_loss = l1(target, identity_target)
            identity_reference_loss = l1(reference, identity_reference)

            # add all togethor
            G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_target_loss * config.LAMBDA_CYCLE
                    + cycle_reference_loss * config.LAMBDA_CYCLE
                    + identity_reference_loss * config.LAMBDA_IDENTITY
                    + identity_target_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:
            os.makedirs(config.SAVE_IMAGE_PATH, exist_ok=True)
            save_image(fake_reference * 0.5 + 0.5, f"{config.SAVE_IMAGE_PATH}/{config.REFERENCE_NAME}_{epoch}_{idx}.png")
            save_image(fake_target * 0.5 + 0.5, f"{config.SAVE_IMAGE_PATH}/{config.TARGET_NAME}_{epoch}_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def tensor2image(tensor):
    if len(tensor.shape) == 4:
        images = []
        for image in tensor:
            img = np.array(transforms.ToPILImage()(image))
            images.append(img)
    return images


def valid_fn(disc_H, disc_Z, gen_Z, gen_H, loader, l1, mse, epoch):
    R_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True, desc=f"{config.DATASET_NAME} TEST, Epoch {epoch}/{config.NUM_EPOCHS}: ")

    for idx, (target, reference) in enumerate(loop):
        target = target.to(config.DEVICE)
        reference = reference.to(config.DEVICE)
        target_img = tensor2image(target)
        reference_img = tensor2image(reference)
        show_destroy_cv2(target_img[0], win_name='target')
        show_destroy_cv2(reference_img[0], win_name='reference')

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_reference = gen_H(target)
            D_R_real = disc_H(reference)
            D_R_fake = disc_H(fake_reference.detach())
            R_reals += D_R_real.mean().item()
            H_fakes += D_R_fake.mean().item()
            D_H_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_H_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_target = gen_Z(reference)
            D_Z_real = disc_Z(target)
            D_Z_fake = disc_Z(fake_target.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 4

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_R_fake = disc_H(fake_reference)
            D_Z_fake = disc_Z(fake_target)
            loss_G_H = mse(D_R_fake, torch.ones_like(D_R_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_target = gen_Z(fake_reference)
            cycle_reference = gen_H(fake_target)
            cycle_target_loss = l1(target, cycle_target)
            cycle_reference_loss = l1(reference, cycle_reference)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_target = gen_Z(target)
            identity_reference = gen_H(reference)
            identity_target_loss = l1(target, identity_target)
            identity_reference_loss = l1(reference, identity_reference)

            # add all togethor
            G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_target_loss * config.LAMBDA_CYCLE
                    + cycle_reference_loss * config.LAMBDA_CYCLE
                    + identity_reference_loss * config.LAMBDA_IDENTITY
                    + identity_target_loss * config.LAMBDA_IDENTITY
            )

        print(f'{epoch}-{idx} D-Loss: {D_loss}, G-Loss: {G_loss}')
        os.makedirs(config.SAVE_IMAGE_PATH, exist_ok=True)
        save_image(fake_reference * 0.5 + 0.5, f"{config.SAVE_IMAGE_PATH}/{config.REFERENCE_NAME}_{epoch}_{idx}.png")
        save_image(fake_target * 0.5 + 0.5, f"{config.SAVE_IMAGE_PATH}/{config.TARGET_NAME}_{epoch}_{idx}.png")

        loop.set_postfix(H_real=R_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    disc_H = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    gen_Z = Generator(img_channels=config.IN_CHANNELS, num_residuals=config.N_BLOCKS).to(config.DEVICE)
    gen_H = Generator(img_channels=config.IN_CHANNELS, num_residuals=config.N_BLOCKS).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = ReferenceTargetDataset(
        root_reference=config.TRAIN_DIR + f"/{config.REFERENCE_NAME}",
        root_target=config.TRAIN_DIR + f"/{config.TARGET_NAME}", transform=config.transforms
    )
    val_dataset = ReferenceTargetDataset(
        root_reference=config.VAL_DIR + f"/{config.REFERENCE_NAME}", root_target=config.TRAIN_DIR + f"/{config.TARGET_NAME}",
        transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    if config.TRAIN:
        for epoch in range(config.NUM_EPOCHS):
            train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)

            if config.SAVE_MODEL:
                save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
                save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
                save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
                save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)
    else:
        for epoch in range(config.TEST_EPOCHS):
            valid_fn(disc_H, disc_Z, gen_Z, gen_H, val_loader, L1, mse, epoch)


if __name__ == "__main__":
    main()
