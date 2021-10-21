import os
from os.path import join

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
import mlflow
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from deep_utils import show_destroy_cv2


def train_fn(disc_R, disc_T, gen_RT, gen_TR, loader, opt_disc, opt_gen, l1, mse, epoch, mlflow):
    disc_R.train()
    disc_T.train()
    gen_TR.train()
    gen_RT.train()
    loop = tqdm(loader, leave=True, desc=f"{config.DATASET_NAME} TRAIN, Epoch {epoch}/{config.NUM_EPOCHS}: ")
    loop_len = len(loop)

    for idx, (target, reference) in enumerate(loop):
        step_num = epoch * len(loop) + idx + 1

        target = target.to(config.DEVICE)
        reference = reference.to(config.DEVICE)

        # Train Discriminators H and Z
        # with torch.cuda.amp.autocast():
        toggle_grad(disc_R, disc_T, True)
        fake_reference = gen_TR(target)
        # d_r results
        D_R_real = disc_R(reference)
        D_R_fake = disc_R(fake_reference.detach())

        # d_r losses
        D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
        D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
        D_R_loss = D_R_real_loss + D_R_fake_loss

        fake_target = gen_RT(reference)

        # d_t results
        D_T_real = disc_T(target)
        D_T_fake = disc_T(fake_target.detach())

        # d_t losses
        D_T_real_loss = mse(D_T_real, torch.ones_like(D_T_real))
        D_T_fake_loss = mse(D_T_fake, torch.zeros_like(D_T_fake))
        D_T_loss = D_T_real_loss + D_T_fake_loss

        # put it together
        D_loss = (D_R_loss + D_T_loss) / 2

        # get the metrics
        # disc + reference loss
        mlflow.log_metric("D_R_real_loss", D_R_real_loss.item(), step=step_num)
        mlflow.log_metric("D_R_fake_loss", D_R_fake_loss.item(), step=step_num)
        mlflow.log_metric("D_R_loss", D_R_loss.item(), step=step_num)
        # disc + target loss
        mlflow.log_metric("D_T_real_loss", D_T_real_loss.item(), step=step_num)
        mlflow.log_metric("D_T_fake_loss", D_T_fake_loss.item(), step=step_num)
        mlflow.log_metric("D_T_loss", D_T_loss.item(), step=step_num)
        # disc loss
        mlflow.log_metric("D_loss", D_loss.item(), step=step_num)

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train Generators H and Z
        # adversarial loss for both generators
        toggle_grad(disc_R, disc_T, False)
        D_R_fake = disc_R(fake_reference)
        D_T_fake = disc_T(fake_target)

        loss_G_TR = mse(D_R_fake, torch.ones_like(D_R_fake))
        loss_G_RT = mse(D_T_fake, torch.ones_like(D_T_fake))

        # cycle loss
        cycle_target = gen_RT(fake_reference)
        cycle_reference = gen_TR(fake_target)
        cycle_target_loss = l1(target, cycle_target) * config.CYCLE_LOSS_COEFFICIENT
        cycle_reference_loss = l1(reference, cycle_reference) * config.CYCLE_LOSS_COEFFICIENT

        # identity loss (remove these for efficiency if you set lambda_identity=0)
        # if config.LAMBDA_IDENTITY:
        #     identity_target = gen_RT(target)
        #     identity_reference = gen_TR(reference)
        #     identity_target_loss = l1(target, identity_target)
        #     identity_reference_loss = l1(reference, identity_reference)
        # else:
        #     identity_target_loss = 0
        #     identity_reference_loss = 0

        # identity losspiq
        # gen_tr_identity = 1 / mse(target, fake_reference) * config.LAMBDA_GEN_IDENTITY
        # gen_rt_identity = 1 / mse(reference, fake_target) * config.LAMBDA_GEN_IDENTITY
        # add all together

        G_loss_execpt_cycle = (
                loss_G_RT +
                loss_G_TR
        )
        cycle_coef = D_loss / G_loss_execpt_cycle
        cycle_coef = cycle_coef.detach()
        cycle_target_loss *= cycle_coef
        cycle_reference_loss *= cycle_coef

        G_loss = (
                loss_G_RT +
                loss_G_TR +
                cycle_target_loss +
                cycle_reference_loss

        )
        # get the metrics
        # gen loss

        mlflow.log_metric('cycle_coef', cycle_coef.item(), step=step_num)
        mlflow.log_metric("loss_G_TR", loss_G_TR.item(), step=step_num)
        mlflow.log_metric("loss_G_RT", loss_G_RT.item(), step=step_num)

        # cycle loss
        mlflow.log_metric("cycle_target_loss", cycle_target_loss.item(), step=step_num)
        mlflow.log_metric("cycle_reference_loss", cycle_reference_loss.item(), step=step_num)

        # identity loss
        # mlflow.log_metric("gen_tr_identity", gen_tr_identity.item(), step=step_num)
        # mlflow.log_metric("gen_rt_identity", gen_rt_identity.item(), step=step_num)
        # final loss
        mlflow.log_metric("G_loss", G_loss.item(), step=step_num)

        os.makedirs(config.SAVE_IMAGE_PATH, exist_ok=True)
        if idx and idx % (loop_len // config.IMG_SAVE_INTERVAL) == 0:
            reference_path = f"{config.SAVE_IMAGE_PATH}/{config.REFERENCE_NAME}_{epoch}_{idx}.png"
            target_path = f"{config.SAVE_IMAGE_PATH}/{config.TARGET_NAME}_{epoch}_{idx}.png"
            save_image(fake_reference * 0.5 + 0.5, reference_path)
            save_image(fake_target * 0.5 + 0.5, target_path)

            # add to mlflow
            mlflow.log_artifact(reference_path, join('images', reference_path))
            mlflow.log_artifact(target_path, join('images', target_path))

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()


def toggle_grad(disc_R, disc_T, grad):
    for net in [disc_R, disc_T]:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = grad


def tensor2image(tensor):
    if len(tensor.shape) == 4:
        images = []
        for image in tensor:
            img = np.array(transforms.ToPILImage()(image))
            images.append(img)
        return images


def valid_fn(disc_R, disc_T, gen_RT, gen_TR, loader, l1, mse, epoch, mlflow):
    disc_R.eval()
    disc_T.eval()
    gen_TR.eval()
    gen_RT.eval()

    loop = tqdm(loader, leave=True, desc=f"{config.DATASET_NAME} TEST, Epoch {epoch}/{config.NUM_EPOCHS}: ")
    loop_len = len(loop)
    for idx, (target, reference) in enumerate(loop):
        step_num = epoch * len(loop) + idx + 1
        target = target.to(config.DEVICE)
        reference = reference.to(config.DEVICE)
        target_img = tensor2image(target)
        reference_img = tensor2image(reference)
        show_destroy_cv2(target_img[0], win_name='target')
        show_destroy_cv2(reference_img[0], win_name='reference')

        # Train Discriminators H and Z
        # with torch.cuda.amp.autocast():
        fake_reference = gen_TR(target)
        D_R_real = disc_R(reference)
        D_R_fake = disc_R(fake_reference.detach())

        D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
        D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
        D_R_loss = D_R_real_loss + D_R_fake_loss

        fake_target = gen_RT(reference)
        D_T_real = disc_T(target)
        D_T_fake = disc_T(fake_target.detach())
        D_T_real_loss = mse(D_T_real, torch.ones_like(D_T_real))
        D_T_fake_loss = mse(D_T_fake, torch.zeros_like(D_T_fake))
        D_T_loss = D_T_real_loss + D_T_fake_loss

        # put it togethor
        D_loss = (D_R_loss + D_T_loss) / 2

        # get the metrics
        # disc + reference loss
        mlflow.log_metric("val_D_R_real_loss", D_R_real_loss.item(), step=step_num)
        mlflow.log_metric("val_D_R_fake_loss", D_R_fake_loss.item(), step=step_num)
        mlflow.log_metric("val_D_R_loss", D_R_loss.item(), step=step_num)
        # disc + target loss
        mlflow.log_metric("val_D_T_real_loss", D_T_real_loss.item(), step=step_num)
        mlflow.log_metric("val_D_T_fake_loss", D_T_fake_loss.item(), step=step_num)
        mlflow.log_metric("val_D_T_loss", D_T_loss.item(), step=step_num)
        # disc loss
        mlflow.log_metric("val_D_loss", D_loss.item(), step=step_num)

        # Train Generators H and Z
        # with torch.cuda.amp.autocast():
        # adversarial loss for both generators
        D_R_fake = disc_R(fake_reference)
        D_T_fake = disc_T(fake_target)
        loss_G_TR = mse(D_R_fake, torch.ones_like(D_R_fake))
        loss_G_RT = mse(D_T_fake, torch.ones_like(D_T_fake))

        # cycle loss
        cycle_target = gen_RT(fake_reference)
        cycle_reference = gen_TR(fake_target)
        cycle_target_loss = l1(target, cycle_target) * config.CYCLE_LOSS_COEFFICIENT
        cycle_reference_loss = l1(reference, cycle_reference) * config.CYCLE_LOSS_COEFFICIENT

        # # identity loss (remove these for efficiency if you set lambda_identity=0)
        # if config.LAMBDA_IDENTITY:
        #     identity_target = gen_RT(target)
        #     identity_reference = gen_TR(reference)
        #     identity_target_loss = l1(target, identity_target)
        #     identity_reference_loss = l1(reference, identity_reference)
        # else:
        #     identity_target_loss = 0
        #     identity_reference_loss = 0

        # identity loss
        # gen_tr_identity = 1 / mse(target, fake_reference) * config.LAMBDA_GEN_IDENTITY
        # gen_rt_identity = 1 / mse(reference, fake_target) * config.LAMBDA_GEN_IDENTITY

        # add all together
        G_loss_execpt_cycle = (
                loss_G_RT +
                loss_G_TR

        )
        cycle_coef = D_loss / G_loss_execpt_cycle
        cycle_coef = cycle_coef.detach()
        cycle_target_loss *= cycle_coef
        cycle_reference_loss *= cycle_coef

        G_loss = (
                loss_G_RT
                + loss_G_TR
                + cycle_target_loss
                + cycle_reference_loss

        )

        # get the metrics
        # gen loss
        mlflow.log_metric("val_loss_G_TR", loss_G_TR.item(), step=step_num)
        mlflow.log_metric("val_loss_G_RT", loss_G_RT.item(), step=step_num)

        # cycle loss
        mlflow.log_metric("val_cycle_target_loss", cycle_target_loss.item(), step=step_num)
        mlflow.log_metric("val_cycle_reference_loss", cycle_reference_loss.item(), step=step_num)

        # identity loss
        # mlflow.log_metric("val_gen_tr_identity", gen_tr_identity.item(), step=step_num)
        # mlflow.log_metric("val_gen_rt_identity", gen_rt_identity.item(), step=step_num)
        # final loss
        mlflow.log_metric("val_G_loss", G_loss.item(), step=step_num)

        print(f'{epoch}-{idx} D-Loss: {D_loss}, G-Loss: {G_loss}')
        os.makedirs(config.SAVE_IMAGE_PATH, exist_ok=True)
        if idx and idx % (loop_len // config.IMG_SAVE_INTERVAL) == 0:
            reference_path = f"{config.SAVE_IMAGE_PATH}/{config.REFERENCE_NAME}_{epoch}_{idx}.png"
            target_path = f"{config.SAVE_IMAGE_PATH}/{config.TARGET_NAME}_{epoch}_{idx}.png"
            save_image(fake_reference * 0.5 + 0.5, reference_path)
            save_image(fake_target * 0.5 + 0.5, target_path)

            # add to mlflow
            mlflow.log_artifact(reference_path, join('images', reference_path))
            mlflow.log_artifact(target_path, join('images', target_path))


def main(mlflow_source='./mlruns', save_dir=''):
    mlflow.set_tracking_uri(mlflow_source)
    mlflow.set_experiment(config.MLFLOW_EXP)
    mlflow.start_run()
    disc_R = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    disc_T = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    gen_RT = Generator(img_channels=config.IN_CHANNELS, num_residuals=config.N_BLOCKS).to(config.DEVICE)
    gen_TR = Generator(img_channels=config.IN_CHANNELS, num_residuals=config.N_BLOCKS).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_R.parameters()) + list(disc_T.parameters()),
        lr=config.DIS_LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_RT.parameters()) + list(gen_TR.parameters()),
        lr=config.GEN_LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            join(save_dir, config.CHECKPOINT_GEN_R), gen_TR, opt_gen, config.GEN_LEARNING_RATE,
        )
        load_checkpoint(
            join(save_dir, config.CHECKPOINT_GEN_T), gen_RT, opt_gen, config.GEN_LEARNING_RATE,
        )
        load_checkpoint(
            join(save_dir, config.CHECKPOINT_DISC_R), disc_R, opt_disc, config.DIS_LEARNING_RATE,
        )
        load_checkpoint(
            join(save_dir, config.CHECKPOINT_DISC_T), disc_T, opt_disc, config.DIS_LEARNING_RATE,
        )

    dataset = ReferenceTargetDataset(
        root_reference=config.TRAIN_DIR + f"/{config.REFERENCE_NAME}",
        root_target=config.TRAIN_DIR + f"/{config.TARGET_NAME}", transform=config.transforms
    )
    val_dataset = ReferenceTargetDataset(
        root_reference=config.VAL_DIR + f"/{config.REFERENCE_NAME}",
        root_target=config.TRAIN_DIR + f"/{config.TARGET_NAME}",
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
    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()
    if config.TRAIN:
        for epoch in range(config.NUM_EPOCHS):
            train_fn(disc_R, disc_T, gen_RT, gen_TR, loader, opt_disc,
                     opt_gen, L1, mse, epoch, mlflow)
            # valid_fn(disc_R, disc_T, gen_RT, gen_TR, val_loader, L1, mse, epoch, mlflow)

            if config.SAVE_MODEL:
                save_checkpoint(gen_TR, opt_gen, filename=join(save_dir, config.CHECKPOINT_GEN_R))
                save_checkpoint(gen_RT, opt_gen, filename=join(save_dir, config.CHECKPOINT_GEN_T))
                save_checkpoint(disc_R, opt_disc, filename=join(save_dir, config.CHECKPOINT_DISC_R))
                save_checkpoint(disc_T, opt_disc, filename=join(save_dir, config.CHECKPOINT_DISC_T))
    else:
        for epoch in range(config.TEST_EPOCHS):
            valid_fn(disc_R, disc_T, gen_RT, gen_TR, val_loader, L1, mse, epoch, mlflow)
    mlflow.end_run()


if __name__ == "__main__":
    main()
