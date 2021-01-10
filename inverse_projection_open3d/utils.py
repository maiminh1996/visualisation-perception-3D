import skimage
import skimage.io
import cv2
import numpy as np

def load_depth_png(depth_path):
    img = skimage.io.imread(depth_path)
    depth = img * 1.0 / 256.0
    # depth = np.reshape(depth, [img.shape[0], img.shape[1], 1]).astype(np.float32)
    depth = np.reshape(depth, [img.shape[0], img.shape[1]]).astype(np.float32)
    return depth

def load_depth_npy(depth_path):
    return np.load(depth_path)

def load_image(img_filename):
    img = cv2.imread(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # left
    return img