import json
import os
import sys
from qt import App
from PyQt5.QtWidgets import QApplication

import cv2
if "mlab" not in sys.modules:
    import mayavi.mlab as mlab

from utils import load_velo_scan
from utils import draw_lidar

def main(lidar_path):    
    pc_velo = load_velo_scan(lidar_path)
    print("haha ", lidar_path)
    draw_lidar(pc_velo) # go to if color is None: color = pc[:, 0] to change color intensity or z
    # mlab.show()
    
if __name__ == '__main__':
    # lidar_path = "../dataset/kitti/training/velodyne_points_beta0.05000/000085.bin"
    # lidar_path = "../dataset/kitti/training/velodyne/000020.bin"
    app = QApplication(sys.argv)
    ex = App()
    lidar_path = ex.initUI()
    main(lidar_path)
    sys.exit(app.exec_())
