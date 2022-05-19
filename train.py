import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config as cfg
from models.loss import RegLoss
from models.mnet import get_mobile_net
from datasets import WiderFace
from utils import Simple_logger


logger = Simple_logger()
writer = SummaryWriter("tb_logs")

logger.log("Setting up")
# Data Setup
trainset = WiderFace(cfg.train_dataroot, cfg.train_annfile, cfg.sigma, cfg.downscale, cfg.insize, cfg.train_transforms)
valset = WiderFace(cfg.val_dataroot, cfg.val_annfile, cfg.sigma, cfg.downscale, cfg.insize, cfg.test_transforms)
trainloader = DataLoader(trainset, batch_size=cfg.batch_size, 
    pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
valloader = DataLoader(valset, batch_size=cfg.batch_size, 
    pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

device = cfg.device

# Network Setup
net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)

# Training Setup
optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
heatmap_loss = nn.MSELoss()
wh_loss = RegLoss()
off_loss = RegLoss()
lm_loss = RegLoss()

# Checkpoints Setup
checkpoints = cfg.checkpoints
os.makedirs(checkpoints, exist_ok=True)

if cfg.restore:
    weights_path = osp.join(checkpoints, cfg.restore_model)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"load weights from checkpoints: {cfg.restore_model}")

# Start training
logger.log("Start training")
net.train()
net.to(device)


for e in range(cfg.epoch):
    logger.log("training for the {} epoch".format(e))
    net.train()
    for data, labels in tqdm(trainloader, desc=f"Epoch {e}/{cfg.epoch}",
                             ascii=True, total=len(trainloader)):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        out = net(data)

        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = heatmap_loss(heatmaps, labels[:, 0])

        offs = torch.cat([o['off'].squeeze() for o in out], dim=0)
        l_off = off_loss(offs, labels[:, [1,2]])

        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = wh_loss(whs, labels[:, [3,4]])

        lms = torch.cat([o['lm'].squeeze() for o in out], dim=0)
        l_lm = lm_loss(lms, labels[:, 5:])

        loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
        loss.backward()
        optimizer.step()
    
    logger.log(f"Epoch {e}/{cfg.epoch}, Train loss, heat: {l_heatmap:.6f}, off: {l_off:.6f}, size: {l_wh:.6f}, landmark: {l_lm:.6f}")
    writer.add_scalar('Train_Loss', loss, e)

    # logger.log("validating for the {} epoch".format(e))
    # net.eval()
    # with torch.no_grad():
    #     for data, labels in tqdm(valloader, desc=f"Epoch {e}/{cfg.epoch}", ascii=True, total=len(valloader)):
    #         data = data.to(device)
    #         labels = labels.to(device)
    #         out = net(data)
    #         heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
    #         l_heatmap = heatmap_loss(heatmaps, labels[:, 0])

    #         offs = torch.cat([o['off'].squeeze() for o in out], dim=0)
    #         l_off = off_loss(offs, labels[:, [1,2]])

    #         whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
    #         l_wh = wh_loss(whs, labels[:, [3,4]])

    #         lms = torch.cat([o['lm'].squeeze() for o in out], dim=0)
    #         l_lm = lm_loss(lms, labels[:, 5:]) 

    #         loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1           
    
    # logger.log(f"Epoch {e}/{cfg.epoch}, Val loss, heat: {l_heatmap:.6f}, off: {l_off:.6f}, size: {l_wh:.6f}, landmark: {l_lm:.6f}")
    # writer.add_scalar('Val_Loss', loss, e)

    backbone_path = osp.join(checkpoints, f"{e}.pth")
    torch.save(net.state_dict(), backbone_path)