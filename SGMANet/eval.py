import torch
import numpy as np

# local/our imports
import pystrum.pynd.ndutils as nd

import ants
import os
import SimpleITK as sitk

import data_loader


def label_to_onehot(label):
    if isinstance(label, str):
        assert os.path.exists(label)

        label = sitk.ReadImage(label)
        label = sitk.GetArrayFromImage(label)
        label = data_loader.onehot_encode(torch.from_numpy(label))
    elif  isinstance(label, torch.Tensor):
        label = label.squeeze()

        if label.ndim == 3:
            label = data_loader.onehot_encode(label)
        elif label.ndim == 4:
            assert(label.shape[0] == 4)
        else:
            raise TypeError('format not support!')
    elif type(label) == ants.core.ants_image.ANTsImage:
        label = label.numpy().transpose((2,1,0))
        label = data_loader.onehot_encode(torch.from_numpy(label))
    elif type(label) == sitk.SimpleITK.Image :
        label = sitk.GetArrayFromImage(label)
        label = data_loader.onehot_encode(torch.from_numpy(label))
    else:
        raise TypeError('format not support!')  
  
    return label 

def disp_def_to_tensor(image):
    if isinstance(image, str):
        assert os.path.exists(image)

        image = sitk.ReadImage(image)
        image = sitk.GetArrayFromImage(image)
        image = image.transpose([3, 0, 1, 2])
        image = torch.from_numpy(image)
    elif  isinstance(image, torch.Tensor):
        image = image.squeeze()
        assert(image.ndim == 4)
        assert(image.shape[0] == 3)
    elif type(image) == ants.core.ants_image.ANTsImage:
        image = image.numpy().transpose((3, 2, 1, 0))
        image = torch.from_numpy(image)
    elif type(image) == sitk.SimpleITK.Image :
        image = sitk.GetArrayFromImage(image).transpose([3, 0, 1, 2])
        image = torch.from_numpy(image)
    else:
        raise TypeError('format not support!') 
    
    return image 


def dsc_one_hot(fixed_label, moved_label):
    ndims = len(list(moved_label.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (fixed_label * moved_label).sum(dim=vol_axes)
    bottom = torch.clamp((fixed_label + moved_label).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom, dim=0)
    return dice

def jacobian_determinant_tensor(disp):

    # check inputs
    disp = disp.permute((1,2,3,0))
    volshape = disp.shape[:-1]

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    dx = J[0]
    dy = J[1]
    dz = J[2]

        # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    
    jcb = Jdet0 - Jdet1 + Jdet2

    return np.sum(jcb <= 0)/np.prod(jcb.shape)

def jacobian_determinant_pytorch(deformation_filed ):
    size = deformation_filed.shape[1:]

    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)

    deformation_filed = grid + deformation_filed

    deformation_filed = deformation_filed.permute((1,2,3,0))

    # compute gradients
    J = torch.gradient(deformation_filed)

    # 3D glow
    dx = J[0]
    dy = J[1]
    dz = J[2]

        # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    
    jcb = Jdet0 - Jdet1 + Jdet2

    jcb_percent = torch.sum(jcb <= 0).item()/np.prod(size)

    return jcb_percent

def jacobian_determinant_from_deformation(deformation_filed ):
    size = deformation_filed.shape[1:]

    deformation_filed = deformation_filed.permute((1,2,3,0))

    # compute gradients
    J = torch.gradient(deformation_filed)

    # 3D glow
    dx = J[0]
    dy = J[1]
    dz = J[2]

        # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    
    jcb = Jdet0 - Jdet1 + Jdet2

    jcb_percent = torch.sum(jcb <= 0).item()/np.prod(size)

    return jcb_percent


# def jacobian_determinant(disp, use_torch=False):
#     if use_torch :
#         jcb_batch = [jacobian_determinant_pytorch(x.cpu()) for x in disp[:]]
#     else:
#         jcb_batch = [jacobian_determinant_tensor(x.cpu()) for x in disp[:]]

#     return np.mean(jcb_batch)

def jacobian_determinant(disp, use_torch=False):
    jcb_batch = [jacobian_determinant_from_deformation(x) for x in disp[:]]

    return np.mean(jcb_batch)