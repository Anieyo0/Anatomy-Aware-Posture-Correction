import torch
import torch.nn.functional as F
import cv2
import numpy as np

def generate_edge_map(img_tensor):
    """
    Parameters:
        img_tensor: [B, 1, H, W] 형식의 0~1 정규화 이미지 텐서
    Returns:
        gradient magnitude map: [B, 1, H, W] tensor
    """
    sobel_x = torch.tensor([[[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]], dtype=img_tensor.dtype).view(1, 1, 3, 3).to(img_tensor.device)
    sobel_y = torch.tensor([[[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]]], dtype=img_tensor.dtype).view(1, 1, 3, 3).to(img_tensor.device)

    grad_x = F.conv2d(img_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(img_tensor, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    
    grad_mag = grad_mag / (grad_mag.max() + 1e-6)

    return grad_mag