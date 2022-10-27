import random
from PIL import Image
import numpy as np
import torch

class AddPepperNoise(object):
    def __init__(self, snr=0.9, p=1.0):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            b, h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(b, h, w, c), p=[signal_pct, noise_pct/2., noise_pct/2.])
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return img_
            
            # return Image.fromarray(img_.astype('uint8'))
        else:
            return img

class AddGaussianNoise(object):
    def __init__(self, p=0.5,mean=0, std=32):
        assert (isinstance(p, float))
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img,dtype=float).copy()
            img_ = img_ / 255.0
            noise = np.random.normal(self.mean, self.std/255.0, img_.shape)
            out = img_ + noise
            resultImg = np.clip(out, 0.0, 1.0)
            resultImg = resultImg * 255.0
            return resultImg.astype('uint8')
            # return Image.fromarray(resultImg.astype('uint8'))
        else:
            return img