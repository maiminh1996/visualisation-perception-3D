import json
import os
import sys

import cv2
if "mlab" not in sys.modules:
    import mayavi.mlab as mlab

from utils import load_velo_scan
from utils import draw_lidar

def main(lidar_path):    
    pc_velo = load_velo_scan(lidar_path)
    draw_lidar(pc_velo) # go to if color is None: color = pc[:, 0] to change color intensity or z
    mlab.show()
    
if __name__ == '__main__':
    lidar_path = "../dataset/kitti/training/velodyne/000020.bin"
    main(lidar_path)
