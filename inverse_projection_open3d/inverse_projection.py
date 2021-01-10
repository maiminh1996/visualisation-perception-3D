import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from geometry_utils import *
from utils import load_depth_png, load_image
import time

def main(img_path, depth_path):
    # load image, depth
    rgb = load_image(img_path)
    depth = load_depth_png(depth_path)
    # Get intrinsic parameters from field of view fov
    height, width, _ = rgb.shape
    K = intrinsic_from_fov(height, width, 90)  # +- 45 degrees
    K_inv = np.linalg.inv(K)
    # Get pixel coordinates
    pixel_coords = pixel_coord_np(width, height)  # [3, npoints]
    # Apply back-projection: K_inv @ pixels * depth
    cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()
    rgb = rgb.reshape((height*width, 3)).transpose()
    # Limit points to 150m in the z-direction for visualisation
    rgb = rgb[:, np.where(cam_coords[2] <= 150)[0]]
    cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]

    def custom_draw_geometry(pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        # opt.background_color = np.asarray([0.1, 0.1, 0.1])
        return False

    # Visualize
    pcd_cam = o3d.geometry.PointCloud()

    # pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
    pcd_cam.points = o3d.Vector3dVector(cam_coords.T[:, :3])
    # Flip it, otherwise the pointcloud will be upside down
    pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd_cam.colors  = o3d.utility.Vector3dVector(rgb.T.astype('float64') / 255.0)
    pcd_cam.colors  = o3d.Vector3dVector(rgb.T.astype('float64') / 255.0)
    
    ### open3d > 0.10.0 ubuntu 18
    # cl, ind = pcd_cam.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # inlier_cloud = pcd_cam.select_by_index(ind)

    #cl1, ind1 = pcd_cam.remove_radius_outlier(nb_points=5, radius=0.5)
    ### open3d 0.7.0 ubuntu 16.04
    # cl, ind = o3d.geometry.statistical_outlier_removal(pcd_cam, nb_neighbors=20, std_ratio=2.0)
    # inlier_cloud = pcd_cam.select_by_index(ind) not found in 0.7.0
    custom_draw_geometry(pcd_cam)
    # o3d.visualization.draw_geometries_with_animation_callback([pcd_cam], change_background_to_black)

def video(img_list, depth_list):
    pcd = o3d.PointCloud()
    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # render_option = vis.get_render_option()
    # render_option.point_size = 0.1
    
    to_reset_view_point = True
    
    Cam_coords = []
    Rgb = []
    
    for i in range(len(img_list)):
        # load image, depth
        rgb = load_image(img_list[i])
        depth = load_depth_png(depth_list[i])
        # Get intrinsic parameters from field of view fov
        height, width, _ = rgb.shape
        K = intrinsic_from_fov(height, width, 90)  # +- 45 degrees
        K_inv = np.linalg.inv(K)
        # Get pixel coordinates
        pixel_coords = pixel_coord_np(width, height)  # [3, npoints]
        # Apply back-projection: K_inv @ pixels * depth
        cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()
        rgb = rgb.reshape((height*width, 3)).transpose()
        # Limit points to 150m in the z-direction for visualisation
        rgb = rgb[:, np.where(cam_coords[2] <= 150)[0]]
        cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]
        Cam_coords.append(cam_coords.T[:, :3])
        Rgb.append(rgb.T.astype('float64') / 255.0)

    while True:
        for i in range(len(img_list)):
            pcd.points = o3d.Vector3dVector(Cam_coords[i])
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcd.colors  = o3d.Vector3dVector(Rgb[i])
            vis.update_geometry()
            if to_reset_view_point:
                vis.reset_view_point(True)
                to_reset_view_point = False
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.2)
    vis.destroy_window()



if __name__ == "__main__":
    # img_path = "../dataset/kitti/raw_data/2011_09_26_drive_0020_sync/image_2/0000000005.png"
    # depth_path = "../dataset/kitti/raw_data/2011_09_26_drive_0020_sync/groundtruth_depth/0000000005.png"
    # img_path = "../dataset/kitti/raw_data/2011_09_28_drive_0106_sync/image_2/0000000048.png"
    # depth_path = "../dataset/kitti/raw_data/2011_09_28_drive_0106_sync/groundtruth_depth/0000000048.png"
    # lidar_path = "../dataset/kitti/raw_data/2011_09_28_drive_0106_sync/velodyne/0000000048.png" # TODO
    img_path = "../dataset/kitti/raw_data/2011_09_28/2011_09_28_drive_0106_sync/image_02/data/0000000048.png"
    depth_path = "../dataset/kitti/raw_data/data_depth_annotated/train/2011_09_28_drive_0106_sync/proj_depth/groundtruth/image_02/0000000048.png"
    main(img_path, depth_path)
    
    # 5-69
    # img_path = "../dataset/kitti/raw_data/2011_09_28_drive_0106_sync/image_2/"
    # depth_path = "../dataset/kitti/raw_data/2011_09_28_drive_0106_sync/groundtruth_depth/"
    img_path = "../dataset/kitti/raw_data/2011_09_28/2011_09_28_drive_0106_sync/image_02/data/"
    depth_path = "../dataset/kitti/raw_data/data_depth_annotated/train/2011_09_28_drive_0106_sync/proj_depth/groundtruth/image_02/"
    img_list = [img_path + '%010d.png' % (i) for i in range(5, 70)] 
    depth_list = [depth_path + '%010d.png' % (i) for i in range(5, 70)] 
    # video(img_list, depth_list)