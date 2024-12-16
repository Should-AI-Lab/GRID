import cv2
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re


def get_frame_number(filename):
    match = re.search(r'frame(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return -1


def concat_images(input_dir, output_dir):
    obj_id = os.path.basename(input_dir)

    image_files = [str(f) for f in Path(input_dir).glob('*.png')]

    image_files.sort(key=get_frame_number)

    if len(image_files) < 24:
        print(f"警告: {input_dir} 中图片数量少于24张")
        return

    # 直接使用ID作为输出文件名
    filename = f"{obj_id}.jpg"

    # 读取第一张图片获取单个图片尺寸
    first_image = cv2.imread(image_files[0])
    h, w = first_image.shape[:2]

    # 创建4x6的大图
    grid = np.zeros((h * 4, w * 6, 3), dtype=np.uint8)

    # 填充图片
    for idx, img_path in enumerate(image_files[:24]):
        row = idx // 6
        col = idx % 6
        img = cv2.imread(img_path)
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存大图
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, grid)
    return filename


def process_all_directories(root_dir, output_dir):
    subdirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]

    print("开始处理所有目录...")
    for subdir in tqdm(subdirs):
        filename = concat_images(str(subdir), output_dir)
        if filename:
            print(f"已保存: {filename}")


# 目录设置
root_directory = "/data3/Cone/Data/4d/diffusiond4d/dynamic4d/"
output_directory = "/data3/Cone/Data/4d/diffusiond4d/concat/"

root_directory = "/data3/Cone/Data/4d/diffusiond4d/static3d/"
output_directory = "/data3/Cone/Data/4d/diffusiond4d/concat-3d/"

root_directory = "/data3/Cone/Data/4d/diffusiond4d/monocular_front/"
output_directory = "/data3/Cone/Data/4d/diffusiond4d/concat-vidfront/"


os.makedirs(output_directory, exist_ok=True)

# 处理所有目录
process_all_directories(root_directory, output_directory)
