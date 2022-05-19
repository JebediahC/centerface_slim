import cv2
import matplotlib.pyplot as plt

from datasets import WiderFace
from config import Config as cfg
widerface = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma, cfg.downscale, cfg.insize, cfg.train_transforms)

im, hm = widerface.__getitem__(10969)
