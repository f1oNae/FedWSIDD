from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


# Gaussian blur kernel
def get_gaussian_kernel(device="cpu"):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
    return gaussian_k


def pyramid_down(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    # channel-wise conv(important)
    multiband = [F.conv2d(image[:, i:i + 1, :, :], gaussian_k, padding=2, stride=2) for i in range(3)]
    down_image = torch.cat(multiband, dim=1)
    return down_image


def pyramid_up(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    upsample = F.interpolate(image, scale_factor=2)
    multiband = [F.conv2d(upsample[:, i:i + 1, :, :], gaussian_k, padding=2) for i in range(3)]
    up_image = torch.cat(multiband, dim=1)
    return up_image


def gaussian_pyramid(original, n_pyramids, device="cpu"):
    x = original
    # pyramid down
    pyramids = [original]
    for i in range(n_pyramids):
        x = pyramid_down(x, device=device)
        pyramids.append(x)
    return pyramids


def laplacian_pyramid(original, n_pyramids, device="cpu"):
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, device=device)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - pyramid_up(pyramids[i + 1], device=device)
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])
    return laplacian


def minibatch_laplacian_pyramid(image, n_pyramids, batch_size, device="cpu"):
    n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size)
    pyramids = []
    for i in range(n):
        x = image[i * batch_size:(i + 1) * batch_size]
        p = laplacian_pyramid(x.to(device), n_pyramids, device=device)
        p = [x.cpu() for x in p]
        pyramids.append(p)
    del x
    result = []
    for i in range(n_pyramids + 1):
        x = []
        for j in range(n):
            x.append(pyramids[j][i])
        result.append(torch.cat(x, dim=0))
    return result


def extract_patches(pyramid_layer, slice_indices,
                    slice_size=7, unfold_batch_size=128, device="cpu"):
    assert pyramid_layer.ndim == 4
    n = pyramid_layer.size(0) // unfold_batch_size + np.sign(pyramid_layer.size(0) % unfold_batch_size)
    # random slice 7x7
    p_slice = []
    for i in range(n):
        # [unfold_batch_size, ch, n_slices, slice_size, slice_size]
        ind_start = i * unfold_batch_size
        ind_end = min((i + 1) * unfold_batch_size, pyramid_layer.size(0))
        x = pyramid_layer[ind_start:ind_end].unfold(
            2, slice_size, 1).unfold(3, slice_size, 1).reshape(
            ind_end - ind_start, pyramid_layer.size(1), -1, slice_size, slice_size)
        # [unfold_batch_size, ch, n_descriptors, slice_size, slice_size]
        x = x[:, :, slice_indices, :, :]
        # [unfold_batch_size, n_descriptors, ch, slice_size, slice_size]
        p_slice.append(x.permute([0, 2, 1, 3, 4]))
    # sliced tensor per layer [batch, n_descriptors, ch, slice_size, slice_size]
    x = torch.cat(p_slice, dim=0)
    # normalize along ch
    std, mean = torch.std_mean(x, dim=(0, 1, 3, 4), keepdim=True)
    x = (x - mean) / (std + 1e-8)
    # reshape to 2rank
    x = x.reshape(-1, 3 * slice_size * slice_size)
    return x


import torch
def rand_projections(dim, num_projections=1000,device='cpu'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance

def ISEBSW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    wasserstein_distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    wasserstein_distances =  wasserstein_distances.view(1,L)
    weights = torch.softmax(wasserstein_distances,dim=1)
    sw = torch.sum(weights*wasserstein_distances,dim=1).mean()
    return  torch.pow(sw,1./p)