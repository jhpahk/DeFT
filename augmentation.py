import torch
import numpy as np
import torchvision.transforms as T
import kornia


Norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # ImageNet statistics

DeNorm = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                      T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]), ])


def SimpleAugRGB(img, iter):
    img_aug = DeNorm(img)
        
    ## global RGB augmentation
    img_mean = img_aug.mean(dim=(-1, -2), keepdim=True)
    img_std = img_aug.std(dim=(-1, -2), keepdim=True) + 1e-7

    img_mean_new = torch.rand_like(img_mean)
    img_std_new = torch.rand_like(img_std)

    img_aug = img_std_new * ((img_aug - img_mean) / img_std) + img_mean_new
    
    ## HFC noise augmentation
    noise_degree = 0.1 + iter * ((0.25 - 0.1) / 10000)
    noise_degree = 0.25 if noise_degree > 0.25 else noise_degree
    hfc_noise = torch.randn_like(img_aug) * noise_degree
    img_aug = torch.clamp(img_aug + hfc_noise, 0, 1)
    img_aug = Norm(img_aug)
    
    output = torch.cat([img, img_aug], dim=0)
    
    return output


def SimpleAugLAB(img, iter):
    img_aug = DeNorm(img)
    
    # RGB to LAB
    img_aug = kornia.color.rgb_to_lab(img_aug)
    
    # normalize to [0, 1]
    norm_factor_shift = torch.tensor([0, 128, 128]).cuda().view(1, -1, 1, 1)
    norm_factor_scale = torch.tensor([100, 255, 255]).cuda().view(1, -1, 1, 1)
    
    img_aug = (img_aug + norm_factor_shift) / norm_factor_scale
        
    ## global RGB augmentation
    img_mean = img_aug.mean(dim=(-1, -2), keepdim=True)
    img_std = img_aug.std(dim=(-1, -2), keepdim=True) + 1e-7

    img_mean_new = 0.2 + 0.6 * torch.rand_like(img_mean)
    img_std_new = 0.1 + 0.3 * torch.rand_like(img_std)

    img_aug = img_std_new * ((img_aug - img_mean) / img_std) + img_mean_new
    
    ## HFC noise augmentation
    # noise_degree = 0.1 + iter * ((0.25 - 0.1) / 10000)
    # noise_degree = 0.25 if noise_degree > 0.25 else noise_degree
    noise_degree = 0.01
    hfc_noise = torch.randn_like(img_aug) * noise_degree
    
    img_aug = torch.clamp(img_aug + hfc_noise, 0, 1)
    img_aug = img_aug * norm_factor_scale - norm_factor_shift
    img_aug = kornia.color.lab_to_rgb(img_aug)
    img_aug = Norm(img_aug)
    
    output = torch.cat([img, img_aug], dim=0)
    
    return output



def rgb2xyz(rgb):
    mask = (rgb > 0.04045).float()
    # rgb = (((rgb + 0.055) / 1.055) ** 2.4) * mask + (rgb / 12.92) * (1 - mask)
    rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    rgb = rgb * 100 # scale to [0, 100]

    # RGB to XYZ conversion matrix
    rgb_to_xyz_matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ]).to(rgb.device)

    xyz = torch.matmul(rgb.permute(0, 2, 3, 1), rgb_to_xyz_matrix.t()).permute(0, 3, 1, 2)
    # xyz = torch.einsum('ij,...jhw->...ihw', rgb_to_xyz_matrix, rgb)
    
    return xyz

def xyz2lab(xyz):
    # Reference white point D65
    xyz_ref_white = torch.tensor([95.047, 100.0, 108.883]).to(xyz.device).view(1, 3, 1, 1)
    xyz = xyz / xyz_ref_white

    epsilon = 0.008856
    kappa = 903.3

    mask = (xyz > epsilon).float()
    # xyz = ((xyz ** (1/3)) * mask) + ((kappa * xyz + 16) / 116) * (1 - mask)
    xyz = torch.where(mask, xyz ** (1/3), (kappa * xyz + 16) / 116)
    xyz = xyz.clamp(min=0)

    L = (116 * xyz[:, 1, :, :]) - 16
    a = 500 * (xyz[:, 0, :, :] - xyz[:, 1, :, :])
    b = 200 * (xyz[:, 1, :, :] - xyz[:, 2, :, :])

    lab = torch.stack([L, a, b], dim=1)
    
    return lab

# input tensor must be normalized to [0, 1]
def rgb2lab(rgb_batch):
    # rgb_batch = rgb_batch / 255.0
    
    xyz_batch = rgb2xyz(rgb_batch)
    lab_batch = xyz2lab(xyz_batch)
    
    return lab_batch

def lab2xyz(lab):
    # Reference white point D65
    xyz_ref_white = torch.tensor([95.047, 100.0, 108.883]).to(lab.device).view(1, 3, 1, 1)
    
    L, a, b = lab[:, 0, :, :], lab[:, 1, :, :], lab[:, 2, :, :]
    
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    
    epsilon = 0.008856  # CIE standard
    kappa = 903.3  # CIE standard
    
    x = torch.where(x ** 3 > epsilon, x ** 3, (116 * x - 16) / kappa)
    y = torch.where(y ** 3 > epsilon, y ** 3, (116 * y - 16) / kappa)
    z = torch.where(z ** 3 > epsilon, z ** 3, (116 * z - 16) / kappa)
    
    x = x.clamp(min=0)
    y = y.clamp(min=0)
    z = z.clamp(min=0)
    
    xyz = torch.stack([x, y, z], dim=1) * xyz_ref_white
    
    return xyz

def xyz2rgb(xyz):
    xyz = xyz / 100.0

    # XYZ to RGB conversion matrix
    xyz_to_rgb_matrix = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ]).to(xyz.device)

    rgb = torch.matmul(xyz.permute(0, 2, 3, 1), xyz_to_rgb_matrix.t()).permute(0, 3, 1, 2)
    rgb = rgb.clamp(min=0)

    mask = (rgb > 0.0031308).float()
    rgb = ((1.055 * (rgb ** (1 / 2.4)) - 0.055) * mask + (rgb * 12.92) * (1 - mask))
    rgb = rgb.clamp(0, 1)
    
    return rgb

# output tensor is normalized to [0, 1]
def lab2rgb(lab_batch):
    xyz_batch = lab2xyz(lab_batch)
    rgb_batch = xyz2rgb(xyz_batch)
    
    return rgb_batch