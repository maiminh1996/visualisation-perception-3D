import json
import os
import sys
import numpy as np
from PIL import Image  # pip install Pillow

import cv2
if "mlab" not in sys.modules:
    import mayavi.mlab as mlab

from utils import load_velo_scan

def main(lidar_path):
    pc_velo = load_velo_scan(lidar_path)
    bev(pc_velo)
    

def scale_to_255(a, min, max, dtype=np.uint8):
    return ((a - min) / float(max - min) * 255).astype(dtype)

def bev(pc_velo):
    
    pointcloud = pc_velo
    # Set the bird's eye view range
    side_range = (-40, 40)  # left and right distance
    fwd_range = (0, 70.4)

    x_points = pointcloud[:, 0]
    y_points = pointcloud[:, 1]
    z_points = pointcloud[:, 2]

    # Get the point in the area
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    x_points = x_points[indices]
    y_points = y_points[indices]

    z_points = z_points[indices]

    res = 0.1  # resolution 0.05m
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # Adjust the origin of the coordinates
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    print(x_img.min(), x_img.max(), y_img.min(), x_img.max())

    # Fill in pixel values
    height_range = (-2, 0.5)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])

    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])

    # Create an image array
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value

    # imshow (Grayscale)
    im2 = Image.fromarray(im)
    im2.show()

    # imshow ( )
    # plt.imshow(im, cmap="nipy_spectral", vmin=0, vmax=255)
    # plt.show()

if __name__ == "__main__":
    lidar_path = "../dataset/kitti/training/velodyne/000020.bin"
    main(lidar_path)