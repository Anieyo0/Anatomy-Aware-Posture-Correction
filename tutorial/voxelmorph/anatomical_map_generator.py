import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_anatomical_map(root_dir, image_size=(256, 256), alpha=0.5, visualize=False):
    """
    여러 강아지 폴더에 있는 모든 이미지로부터 하나의 anatomical map 생성 (mean_std 기반)

    Parameters:
        root_dir (str): 강아지별 이미지 폴더들이 들어있는 상위 폴더
        image_size (tuple): resize할 크기
        alpha (float): softening 계수 (std에 적용)
        visualize (bool): anatomical map 시각화 여부

    Returns:
        anatomical_map (np.ndarray): shape = (H, W), 값 범위 [0, 1]
    """
    # 모든 하위 폴더를 순회하며 이미지 경로 수집
    all_img_paths = []
    for subdir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subdir)
        if os.path.isdir(sub_path):
            all_img_paths += sorted(
                glob(os.path.join(sub_path, '*.jpg')) +
                glob(os.path.join(sub_path, '*.png'))
            )

    if len(all_img_paths) == 0:
        raise FileNotFoundError(f"No images found under {root_dir}")

    imgs = []
    for path in all_img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)
        imgs.append(img.astype(np.float32))  # float으로 변환

    imgs = np.stack(imgs, axis=0)  # shape: (N, H, W)

    # mean map 계산
    mean_map = np.mean(imgs, axis=0)
    mean_scaled = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-6)

    # std map 계산
    std_map = np.std(imgs, axis=0)
    std_map_normalized = std_map / (std_map.max() + 1e-6)

    # mean_std 기반 anatomical map
    anatomical_map = (1 - np.power(std_map_normalized, alpha)) * mean_scaled

    # 값 범위 clip (0~1 보장)
    anatomical_map = np.clip(anatomical_map, 0, 1)

    # 시각화
    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(anatomical_map, cmap='hot', vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Anatomical Map (mean_std based)')
        plt.axis('off')
        plt.show()

    return anatomical_map
