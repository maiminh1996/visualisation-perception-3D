import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from geometry_utils import *
from utils import load_depth_png, load_image, Calibration, Calibration0
from utils import load_from_bin, transform_original_image
import time, os, sys
from tqdm import tqdm

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.append(os.path.join(MY_DIRNAME))



from visualize_kitti.utils import get_lidar_in_image_fov
# from visualize_kitti.utils import Calibration

def show_depth(depth):
    from PIL import Image
    from matplotlib import cm
    import matplotlib.pyplot as plt
    im = Image.fromarray(np.uint8(depth))  # .convert('RGB')
    # im = Image.fromarray(np.uint8(cm.gist_earth(depth_map)))
    # im.save('/home/zelt/Desktop/books_read.png')
    plt.imshow(im)# , vmin=0, vmax=5)
    # plt.imsave('/home/zelt/Desktop/test/4_beams_right.png', im, vmin=0, vmax=30)  # , cmap='hot')
    plt.show()

def main(img_path, depth_path):
    # load image, depth
    rgb = load_image(img_path)
    rgb = transform_original_image(rgb)
    depth = load_depth_png(depth_path)

    # show_depth(depth)
    
    print(np.count_nonzero(depth!=0))
    
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
    # rgb = rgb[:, np.where(cam_coords[1] >= 1.8)[0]]
    # cam_coords = cam_coords[:, np.where(cam_coords[1] >= 1.8)[0]]
    # Limit points to 3m hauteur in the x-direction for visualisation 
    rgb = rgb[:, np.where(cam_coords[1] >= -1)[0]] 
    cam_coords = cam_coords[:, np.where(cam_coords[1] >= -1)[0]]

    def custom_draw_geometry(pcd, fov_step):
        # http://www.open3d.org/docs/0.9.0/tutorial/Advanced/customized_visualization.html
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        render_option = vis.get_render_option()
        render_option.point_size = 0.1
        
        vis.add_geometry(pcd)

        ctr = vis.get_view_control()
        # print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
        ctr.change_field_of_view(step=fov_step)
        
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
    custom_draw_geometry(pcd_cam, 90.0)
    # o3d.visualization.draw_geometries_with_animation_callback([pcd_cam], change_background_to_black)

def sparse0(img_path, lidar_path, calib_path):
    # load image, depth
    rgb = load_image(img_path)
    img_height, img_width, img_channel = rgb.shape
    calib = Calibration0(calib_path)
    # depth = load_depth_png(lidar_path)
    pc_velo = load_from_bin(lidar_path)
    ######################
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo[:, :3], calib, 0, 0, img_width, img_height, True)

    imgfov_pts_2d = pts_2d[fov_inds, :]
    print(imgfov_pts_2d.shape)

    # project those 3d pcl in velo coord to rect cam coord
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255


    img_test = np.zeros([rgb.shape[0], rgb.shape[1]])

    for i in tqdm(range(imgfov_pts_2d.shape[0])):
        depth = imgfov_pc_rect[i, 2]
        # print(img_test.shape)
        # print("{}, {}".format(int(np.round(imgfov_pts_2d[i, 1])), int(np.round(imgfov_pts_2d[i, 0]))))
        img_test[int(np.round(imgfov_pts_2d[i, 1]))-1, int(np.round(imgfov_pts_2d[i, 0]))-1] = depth
    depth = img_test
    print(np.count_nonzero(depth!=0))
    
    
    from PIL import Image
    from matplotlib import cm
    import matplotlib.pyplot as plt
    im = Image.fromarray(np.uint8(img_test))  # .convert('RGB')
    # im = Image.fromarray(np.uint8(cm.gist_earth(depth_map)))
    # im.save('/home/zelt/Desktop/books_read.png')
    plt.imshow(im, vmin=0, vmax=5)
    # plt.imsave('/home/zelt/Desktop/test/4_beams_right.png', im, vmin=0, vmax=30)  # , cmap='hot')
    plt.show()
    
    ######################        
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
        render_option = vis.get_render_option()
        # render_option.point_size = 0.01
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

def sparse(img_path, lidar_path, velo_to_cam, cam_to_cam):
    # load image, depth
    rgb = load_image(img_path)
    img_height, img_width, img_channel = rgb.shape
    calib = Calibration(velo_to_cam, cam_to_cam)
    # depth = load_depth_png(lidar_path)
    pc_velo = load_from_bin(lidar_path)
    ######################
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo[:, :3], calib, 0, 0, img_width, img_height, True)

    imgfov_pts_2d = pts_2d[fov_inds, :]
    print(imgfov_pts_2d.shape)

    # project those 3d pcl in velo coord to rect cam coord
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255


    img_test = np.zeros([rgb.shape[0], rgb.shape[1]])

    for i in tqdm(range(imgfov_pts_2d.shape[0])):
        depth = imgfov_pc_rect[i, 2]
        # print(img_test.shape)
        # print("{}, {}".format(int(np.round(imgfov_pts_2d[i, 1])), int(np.round(imgfov_pts_2d[i, 0]))))
        img_test[int(np.round(imgfov_pts_2d[i, 1]))-1, int(np.round(imgfov_pts_2d[i, 0]))-1] = depth
    depth = img_test
    print(np.count_nonzero(depth!=0))
    
    
    from PIL import Image
    from matplotlib import cm
    import matplotlib.pyplot as plt
    im = Image.fromarray(np.uint8(img_test))  # .convert('RGB')
    # im = Image.fromarray(np.uint8(cm.gist_earth(depth_map)))
    # im.save('/home/zelt/Desktop/books_read.png')
    plt.imshow(im, vmin=0, vmax=5)
    # plt.imsave('/home/zelt/Desktop/test/4_beams_right.png', im, vmin=0, vmax=30)  # , cmap='hot')
    plt.show()
    
    ######################        
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
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.add_geometry(pcd)
    # render_option = vis.get_render_option()
    # render_option.point_size = 0.1
    
    to_reset_view_point = True
    
    Cam_coords = []
    Rgb = []
    
    for i in tqdm(range(len(img_list))):
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
            time.sleep(0.1)
    vis.destroy_window()

def video_sparse(img_list, lidar_list, velo_to_cam, cam_to_cam):
    pcd = o3d.PointCloud()
    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # render_option = vis.get_render_option()
    # render_option.point_size = 0.1
    
    to_reset_view_point = True
    
    Cam_coords = []
    Rgb = []
    
    for i in tqdm(range(len(img_list))):
        # load image, depth
        rgb = load_image(img_list[i])


        img_height, img_width, img_channel = rgb.shape
        calib = Calibration(velo_to_cam, cam_to_cam)
        # depth = load_depth_png(lidar_path)
        pc_velo = load_from_bin(lidar_list[i])
        ######################
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo[:, :3], calib, 0, 0, img_width, img_height, True)

        imgfov_pts_2d = pts_2d[fov_inds, :]
        # print(imgfov_pts_2d.shape)

        # project those 3d pcl in velo coord to rect cam coord
        imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

        import matplotlib.pyplot as plt

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255


        img_test = np.zeros([rgb.shape[0], rgb.shape[1]])

        for i in range(imgfov_pts_2d.shape[0]):
            depth = imgfov_pc_rect[i, 2]
            # print(img_test.shape)
            # print("{}, {}".format(int(np.round(imgfov_pts_2d[i, 1])), int(np.round(imgfov_pts_2d[i, 0]))))
            img_test[int(np.round(imgfov_pts_2d[i, 1]))-1, int(np.round(imgfov_pts_2d[i, 0]))-1] = depth
        depth = img_test


        # depth = load_depth_png(depth_list[i])
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
    lidar_path = "../dataset/kitti/raw_data/2011_09_28/2011_09_28_drive_0106_sync/velodyne_points/data/0000000048.bin"
    depth_path = "../dataset/kitti/raw_data/data_depth_annotated/train/2011_09_28_drive_0106_sync/proj_depth/groundtruth/image_02/0000000048.png"
    calib_cam_to_cam = "../dataset/kitti/raw_data/2011_09_28/calib_cam_to_cam.txt"
    calib_velo_to_cam = "../dataset/kitti/raw_data/2011_09_28/calib_velo_to_cam.txt"
    # main(img_path, depth_path)
    # sparse(img_path, lidar_path, calib_velo_to_cam, calib_cam_to_cam)
    
    # 5-69
    # img_path = "../dataset/kitti/raw_data/2011_09_28_drive_0106_sync/image_2/"
    # depth_path = "../dataset/kitti/raw_data/2011_09_28_drive_0106_sync/groundtruth_depth/"
    img_path = "../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0032_sync/image_02/data/"
    lidar_path = "../dataset/kitti/raw_data/2011_09_26/2011_09_26_drive_0032_sync/velodyne_points/data/"
    depth_path = "../dataset/kitti/raw_data/data_depth_annotated/train/2011_09_26_drive_0032_sync/proj_depth/groundtruth/image_02/"
    img_list = [img_path + '%010d.png' % (i) for i in range(5, 375)] 
    lidar_list = [lidar_path + '%010d.bin' % (i) for i in range(5, 375)] 
    depth_list = [depth_path + '%010d.png' % (i) for i in range(5, 375)] 
    # video(img_list, depth_list)
    # video_sparse(img_list, lidar_list, calib_velo_to_cam, calib_cam_to_cam)

    img_path = "../dataset/kitti/training/image_2/000085.png"
    # img_path = "../dataset/kitti/training/image_beta0.03745/000085.png"
    depth_path = "../dataset/kitti/training/depth_spaded/000020.png"
    
    lidar_path = "../dataset/kitti/training/velodyne/000085.bin"
    # lidar_path = "../dataset/kitti/training/velodyne_points_beta0.03745/000085.bin"
    calib_path = "../dataset/kitti/training/calib/000085.txt"
    # main(img_path, depth_path)
    sparse0(img_path, lidar_path, calib_path)