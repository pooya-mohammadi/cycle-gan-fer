import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "fer"
TRAIN_DIR = f"datasets/{DATASET_NAME}/train"
VAL_DIR = f"datasets/{DATASET_NAME}/test"
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
N_BLOCKS = 6
IN_CHANNELS = 1
NUM_WORKERS = 2
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
REFERENCE_NAME = 'neutral'
TARGET_NAME = 'disgust'
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"
SAVE_IMAGE_PATH = 'save_images'
IMG_w, IMG_H = 48, 48
MEAN, STD = [0.5], [0.5]
transforms = A.Compose(
    [
        A.Resize(width=IMG_w, height=IMG_H),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
