import random

import torch

import numpy as np
import math
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt


def mat2npy(mat, **kwargs):
    img = mat['data']
    ri = mat['ri']
    return img, ri

def calibration(img, ri):
    img = (img - ri) * 9
    img[img < 0] = 0
    return img

def random_crop_3d(img, target_shape=(96, 96), z_start=(64-48)//2, z_end=(64-48)//2+48,**kwargs): # z_start=10, z_end=30
    origin_size = img.shape
    #print(origin_size)
    rand_1 = random.randint(0, origin_size[0]-target_shape[0])
    rand_2 = random.randint(0, origin_size[1]-target_shape[1])
    img = img[rand_1:rand_1+target_shape[0], rand_2:rand_2+target_shape[1], z_start:z_end]
    return img

def multi_crop_3d(img, t_s=(96, 96), z_start=(64-48)//2, z_end=(64-48)//2+48, **kwargs):
    o_s = img.shape
    x = random.randint(1, o_s[0]-t_s[0]-1)
    y = random.randint(1, o_s[1]-t_s[1]-1)
    img1 = img[x:x+t_s[0], y:y+t_s[1], z_start:z_end]
    img2 = img[x-1:x+t_s[0]-1, y:y+t_s[1], z_start:z_end]
    img3 = img[x+1:x+t_s[0]+1, y:y+t_s[1], z_start:z_end]
    img4 = img[x:x+t_s[0], y-1:y+t_s[1]-1, z_start:z_end]
    img5 = img[x:x+t_s[0], y+1:y+t_s[1]+1, z_start:z_end]
    return [img1, img2, img3, img4, img5]


def center_crop_3d(img, target_shape=[96, 96], z_start=(64-48)//2, z_end=(64-48)//2+48, **kwargs):
    origin_size = img.shape
    middle = origin_size[0]//2
    half = target_shape[0]//2
    img = img[middle-half:middle+half,middle-half:middle+half, z_start:z_end]
    return img

def center_multi_crop_3d(img, t_s=(96, 96), z_start=(64-48)//2, z_end=(64-48)//2+48, **kwargs):
    o_s = img.shape
    s = (o_s[0] // 2) - (t_s[0] // 2)
    e = (o_s[0] // 2) + (t_s[0] // 2)
    img1 = img[s:e, s:e, z_start:z_end]
    img2 = img[s-1:e-1, s:e, z_start:z_end]
    img3 = img[s+1:e+1, s:e, z_start:z_end]
    img4 = img[s:e, s-1:e-1, z_start:z_end]
    img5 = img[s:e, s+1:e+1, z_start:z_end]
    return [img1, img2, img3, img4, img5]

def gaussian_3d(img, **kwargs):
    sigma = 0.001#random.uniform(0.01,0.05)
    noise = np.random.normal(0, sigma, size=img.shape)

    if isinstance(img, list):
        return [i + noise for i in img]

    return img + noise


def flipud_3d(img, **kwargs):
    rand = random.randint(0,1)
    if isinstance(img, list):
        if rand == 0:
            return [i[::-1, :, :].copy() for i in img]
        else:
            return img

    if rand==0:
        return img[::-1,:,:].copy()
    else:
        return img

                


def fliplr_3d(img, **kwargs):
    rand = random.randint(0,1)
    if isinstance(img, list):
        if rand == 0:
            return [i[:, ::-1, :].copy() for i in img]
        else:
            return img

    if rand ==0:
        return img[:,::-1,:].copy()
    else:
        return img

        
        
        
def flipud_2d(img, **kwargs): # can't do this due to lack of flip symmetry of f_dx, f_dy
    return img
        
def fliplr_2d(img, **kwargs): # can't do this due to lack of flip symmetry of f_dx, f_dy
    return img
    
    
    
# https://github.com/scipy/scipy/issues/5925
def rotate_3d(img, **kwargs):
    angle = random.randrange(0,360,15)
    if isinstance(img, list):
        return [ndimage.interpolation.rotate(i, angle,
                                        reshape=False,
                                        order=0,
                                        mode='constant', cval = 1.3374) for i in img]

    #rand = random.randint(1,360)
    return ndimage.interpolation.rotate(img, angle,
                                        reshape=False,
                                        order=0,
                                        mode='constant', cval = 1.3374)



def rotate_2d(img, **kwargs):
    angle = random.randrange(0,360,15)
    angle_incre = 7.5
    if isinstance(img, list):
        idx_obph = list(range(1,img_new[0].shape[2]//2))
        idx_obph = np.append(0, np.roll(idx_obph,-int(angle/angle_incre)))
        idx_obamp = list(range(img_new[0].shape[2]//2+1,img_new[0].shape[2]))
        idx_obamp = np.append(img_new[0].shape[2]//2, np.roll(idx_obamp,-int(angle/angle_incre)))
        
        img_new = [i[:,:,np.append(idx_obph, idx_obamp)] for i in img]

        img_new = [np.stack(ndimage.interpolation.rotate(i[:,:,0:(i.shape[2]/2)], angle,reshape=False,order=0,mode='reflect', cval = 0),
                    ndimage.interpolation.rotate(i[:,:,(i.shape[2]/2):i.shape[2]], angle,reshape=False,order=0,mode='reflect', cval = 1), axis = 2)  for i in img]
        

        
        return img_new

    idx_obph = list(range(1,img_new.shape[2]//2))
    idx_obph = np.append(0, np.roll(idx_obph,-int(angle/angle_incre)))
    idx_obamp = list(range(img_new.shape[2]//2+1,img_new.shape[2]))
    idx_obamp = np.append(img_new.shape[2]//2, np.roll(idx_obamp,-int(angle/angle_incre)))
        
    img_new = img[:,:,np.append(idx_obph, idx_obamp)]
    
    img_new = np.stack(ndimage.interpolation.rotate(i[:,:,0:(i.shape[2]/2)], angle,reshape=False,order=0,mode='reflect', cval = 0),
        ndimage.interpolation.rotate(i[:,:,(i.shape[2]/2):i.shape[2]], angle,reshape=False,order=0,mode='reflect', cval = 1), axis = 2)
    return img_new

def to_tensor(img, **kwargs):
    if isinstance(img, list):
        img = np.stack(img, axis=0)
    
    out = torch.from_numpy(img).float()
    if len(out.shape) == 3:
        out = out.unsqueeze(0)
    return out


def to_tensor_2d(img, **kwargs):
    if isinstance(img, list):
        img = np.stack(img, axis=0)
    out = torch.from_numpy(img).float()
    if len(out.shape) == 2: #2d
        out = out.unsqueeze(0)
    else: # multiview
        out = np.transpose(out, (2,0,1))
    return out
    
    
# (1, 1), (5, 2), (1, 0.5), (1, 3)
def elastic_transform(img, alpha=0, sigma=0, random_state=None, **kwargs):

    param_list = [(1, 1), (5, 2), (1, 0.5), (1, 3)]
    if alpha == 0 and sigma == 0:
        rand = random.randint(0,3)
        alpha, sigma = param_list[rand]

    if random_state is None:
       random_state = np.random.RandomState(None)

    shape = img.shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #print(np.mean(dx), np.std(dx), np.min(dx), np.max(dx))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    if isinstance(img, list):
        new_imgs = []
        for i in img:
            new_imgs.append(elastic_transform(i, alpha=alpha, sigma=sigma))
        return new_imgs

    trasform_img = np.zeros(img.shape)
    for i in range(img.shape[2]):
        trasform_img[:, :, i] = map_coordinates(img[:, :, i], indices, order=1).reshape(shape)

    return trasform_img

def binary_mask(img, **kwargs):#
    threshold = 1.3425#math.pi/20#
    img = (img > threshold).astype(float)#
    return img#




class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, ri=None):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W, Z).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        H = img.size(1)
        W = img.size(2)
        Z = img.size(3)

        mask = np.ones((H, W, Z), np.float32)

        for n in range(self.n_holes):
            z = np.random.randint(Z)
            y = np.random.randint(H)
            x = np.random.randint(W)

            y1 = np.clip(y - self.length // 2, 0, H)
            y2 = np.clip(y + self.length // 2, 0, H)
            x1 = np.clip(x - self.length // 2, 0, W)
            x2 = np.clip(x + self.length // 2, 0, W)
            z1 = np.clip(z - self.length // 2, 0, Z)
            z2 = np.clip(z + self.length // 2, 0, Z)

            mask[y1: y2, x1: x2, z1:z2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


TRAIN_AUGS_3D = [
    # multi_crop_3d,
    gaussian_3d,
    elastic_transform,
    flipud_3d,
    fliplr_3d,
    rotate_3d,
    random_crop_3d,
    # binary_mask,#
    to_tensor,
    # Cutout(3, 8)
]

TEST_AUGS_3D = [
    center_crop_3d,
    # center_multi_crop_3d,
    # binary_mask,
    to_tensor
]

if __name__ == "__main__":
    path = "/data2/DW/180930_bac/181017_Bacteria/train/Smarcescens/Tomo_532nm_suppr_1_sp192_timeMarker_001_data.mat.mat"

    import scipy.io as io
    mat = io.loadmat(path)
    img, ri = mat2npy(mat)
    for t in TRAIN_AUGS_3D:
        img = t(img, ri=ri)
        try:
            print(t, img.shape)
        except:
            print(t, len(img))