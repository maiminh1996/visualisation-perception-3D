import json
import os
import sys

import cv2
if "mlab" not in sys.modules:
    import mayavi.mlab as mlab

from utils import kitti_object
from utils import show_image_with_boxes, show_lidar_with_boxes
from utils import show_lidar_on_image, show_lidar_topview_with_boxes

from  utils import draw_lidar_simple, draw_lidar

def main(path, data_idx):
    """
    draw_lidar_simple()
    draw_lidar()
    show_image_with_boxes():
    show_lidar_with_boxes():
    show_lidar_topview_with_boxes():
    show_lidar_on_image():
    :return:
    """
    dataset = kitti_object(path)
    
    print("--------------- ", data_idx)
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()  # WARNING: print only 1st object information in each image

    img = dataset.get_image(data_idx)
    # img = dataset.get_image_right(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # left
    # img = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB) # right
    img_height, img_width, img_channel = img.shape
    calib = dataset.get_calibration(data_idx)
    pc_velo = dataset.get_lidar(data_idx)  # (nx4) with 4: x,y,z,r

    draw_lidar_simple(pc_velo[:,0:3])
    # draw_lidar(pc_velo) # go to if color is None: color = pc[:, 0] to change color intensity or z
    # show_lidar_with_boxes(pc_velo[:, 0:3], objects, calib, True, img_width, img_height)
    # show_image_with_boxes(img, objects, calib, True)
    # show_lidar_topview_with_boxes(pc_velo, objects, calib)  # TODO recode and understand
    # show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height, side='left')

if __name__ == '__main__':
    dataset_path = "../dataset/kitti"
    data_idx = 20 #85: #20, 31, 47
    main(dataset_path, data_idx)
