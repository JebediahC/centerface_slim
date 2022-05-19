import torch
from torchvision import transforms as T


class Config:
    # preprocess
    insize = [416, 416] # 输入尺寸
    channels = 3
    downscale = 4 #下采样
    sigma = 2.65

    train_transforms = T.Compose([
        T.ColorJitter(0.5, 0.5, 0.5, 0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    # dataset
    train_dataroot = 'data/WIDER_train/WIDER_train/images'
    train_annfile = 'data/retinaface_gt_v1.1_adjusted/train/label.txt'

    val_dataroot = 'data/WIDER_val/WIDER_val/images'
    val_annfile = 'data/retinaface_gt_v1.1_adjusted/val/label.txt'

    # checkpoints
    checkpoints = 'checkpoints'
    restore = False
    restore_model = 'final.pth'

    # training
    epoch = 100
    lr = 5e-4
    batch_size = 4
    pin_memory = True
    num_workers = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # inference
    threshold = 0.5
