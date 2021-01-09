import os, sys

import cv2
import numpy as np

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
sys.path.append(os.path.join(MY_DIRNAME))

import kitti_util as utils
from iou_3d import box3d_iou as iou3d
import mayavi.mlab as mlab

np.random.seed(0)

# Create the kitti object class
class kitti_object(object):
    """
    Load and parse object data for 3D object detection into a usable format
    """

    def __init__(self, root_dir, split='training', right_image=True):
        """
        Initialize some directory and traning or testing mode
        :param root_dir: contains training and testing folders for 3D OD
        :param split: flag for split training data into a new training set and a validation set
        """
        self.root_dir = root_dir
        self.split = split  # training flag
        self.split_dir = os.path.join(root_dir, split)  # get training folder

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        if right_image==True:
            self.image_dir_right = os.path.join(self.split_dir, 'image_3') #right
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        # self.pseudo_lidar_dir = os.path.join("/home/zelt/depth_completion/DeepLidar_rewrite/DeepLiDAR/pseudolidar_predict_result")
        
        if split == 'training':
            self.label_dir = os.path.join(self.split_dir, 'label_2')
            # self.num_samples = len(os.listdir(self.image_dir))
            # assert self.num_samples == 7481, "Number of training set needs to be 7481!"
            self.num_samples = 7481
        elif split == 'testing':
            # self.num_samples = len(os.listdir(self.image_dir))
            # assert self.num_samples == 7518, "Number of training set needs to be 7518!"
            self.num_samples = 7518
        else:
            print("Unknown split: ", split)
            exit(-1)




    def get_image(self, idx):
        """

        :param idx:
        :return:
        """
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_image_right(self, idx):
        """

        :param idx:
        :return:
        """
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir_right, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        """

        :param idx:
        :return:(n, 4), n point cloud, 4: x, y, z, reflectance values
        """
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return utils.load_velo_scan(lidar_filename)

    def get_pseudo_lidar(self, idx):
        """

        :param idx:
        :return:(n, 4), n point cloud, 4: x, y, z, reflectance values
        """
        assert (idx < self.num_samples)
        # lidar_filename = os.path.join(self.pseudo_lidar_dir, '%06d.bin' % (idx))
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        """
        Get calib object
        :param idx: for calib file name
        :return:
        """
        assert (idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        # calib_filename = "/media/zelt/Données/KITTI/raw/calib/2011_09_26/2011_09_26.txt"

        # calib_filename = "/media/zelt/Données/KITTI/raw/calib/2011_09_28/2011_09_28.txt"
        # calib_filename = "/media/zelt/Données/KITTI/raw/calib/2011_09_26/2011_09_26.txt"
        # calib_filename = "/media/zelt/Données/KITTI/raw/calib/2011_09_26/2011_09_26.txt"
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        """

        :param idx:
        :return:
        """
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))  # label_filename format like 000001.txt
        return utils.read_label(label_filename)

    def __len__(self):
        """
        Usage return len(kittiObject)
        :return: len(kittiObject)
        """
        return self.num_samples

class kitti_object_add_prediction(kitti_object):
    def __init__(self, root_dir, root_pred, split='training'):
        super().__init__(root_dir, split='training')
        self.root_pred = root_pred
    def get_predict_objects(self, idx):
        # assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.root_pred, '%06d.txt' % (idx))  # label_filename format like 000001.txt
        return utils.read_label(label_filename)

def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    print("top_image:", top_image.shape)

    # gt

    def bbox3d(obj):
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    cv2.imshow("top_image", top_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_with_boxes(img, objects, calib, show3d=True):
    """
    Show image with 2D bounding boxes
    :param img: image loaded
    :param objects:
    :param calib:
    :param show3d:
    :return:
    """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:  # process each object in an image
        if obj.type == 'DontCare': continue  # remove 'DontCare' class
        # draw 2d bbox
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), (0, 255, 0), 2)

        # calculate 3d bbox for left color cam from P: P2
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        print("********************, ", obj.type)
        # print("********************, ", box3d_pts_2d.shape)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)

    #  Image.fromarray(img1).show()
    cv2.imshow('2D bounding box in image', cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    if show3d:
        # Image.fromarray(img2).show()
        cv2.imshow('3D bounding box in image', cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0, depth=False, side='left'):
    """
    Filter lidar points, keep those in image FOV
    :param pc_velo:
    :param calib:
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :param return_more:
    :param clip_distance:
    :return:
    image2 coord:
    (0,0) ----> x-axis (u) x-max = img_height
    |
    |
    v y-axis (v)
    y-max = img_width
    """
    # Project velo point cloud to image:
    # velo --> ref --> rect --> image
    if side=='left':
        pts_2d = calib.project_velo_to_image(pc_velo, depth=depth)
    else:
        pts_2d = calib.project_velo_to_image(pc_velo, depth=depth, side=side)
    # Filter lidar pcl, keep those in image FOV
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo




def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
    data_idx=None,
    pseudo_lidar=None
    ):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None,
        # bgcolor=(1, 1, 1), # black background 
        bgcolor=(0, 0, 0), # white background
        fgcolor=None, engine=None, size=(1000, 500)
        # figure=None
    )

    if img_fov: # filter pcl out of fov
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(pc_velo.shape)
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pts_color=(0,1,0))
    
    # show_box_lidar(objects, calib, data_idx, fig)

    car_obj = []
    color = (1, 0, 0)

    for obj in objects:
        if obj.type == "DontCare":
            print("############## Dont care gt: t: {}, (h: {}, l: {}, w: {})".format(obj.t, obj.h, obj.l, obj.w))
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

        if obj.type == "Car":
            car_obj.append(box3d_pts_3d)
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, data_idx=data_idx, type=obj.type, occlu=obj.occlusion)

        # Draw depth from ego-vehicle to objects
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))

        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    mlab.show()


def find_false_positives(
    pc_velo,
    objects,
    calib,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
    data_idx=None
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d, draw_gt_boxes3d_new

    # print(("All point num: ", pc_velo.shape[0]))


    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        # print(("FOV point num: ", pc_velo.shape[0]))
    # print("pc_velo", pc_velo.shape)
    # draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]
    # """
    car_obj = []
    color = (1, 0, 0)
    for obj in objects:
        if obj.type == "DontCare":
            # print("############## Dont care gt: t: {}, (h: {}, l: {}, w: {})".format(obj.t, obj.h, obj.l, obj.w))
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # print("box3d_pts_3d_velo:")
        # print(box3d_pts_3d_velo)
        if obj.type == "Car":
            car_obj.append(box3d_pts_3d)
        # draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, data_idx=data_idx, type=obj.type, occlu=obj.occlusion)

        # Draw depth
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            # print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            # print("dep_pc_velo:", dep_pc_velo)

    """
    if objects_pred is not None:
        color = (0, 1, 0)
        mask_fps = []
        for obj in objects_pred:
            if obj.type == "DontCare":
                print("############## Dont care pred: t: {}, (h: {}, l: {}, w: {})".format(obj.t, obj.h, obj.l, obj.w))
                continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            # calculate iou
            IoU3D = []
            for car_truth in car_obj:
                IoU3D.append(iou3d(car_truth, box3d_pts_3d)[0])
                # print("----------------------------IoU3D: ", IoU3D[0])
            if len(IoU3D)==0: IoU3D=[0, 0]
            # print("box3d_pts_3d_velo:")
            # print(box3d_pts_3d_velo)
            # draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, iou3d=max(IoU3D))
            false_positive = draw_gt_boxes3d_new([box3d_pts_3d_velo], color=color, iou3d=max(IoU3D))
            mask_fps.append(false_positive)
    """
    iou_max = []
    for h in objects:
        if h.type == "DontCare":
            # print("############## Dont care gt: t: {}, (h: {}, l: {}, w: {})".format(obj.t, obj.h, obj.l, obj.w))
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(h, calib.P)

        if h.type == "Car":
            # if (max(box3d_pts_2d[:, 1]) - min(box3d_pts_2d[:, 1])) < 25:
            #     continue
            for k in objects_pred:
                if k.type == "DontCare":
                    continue
                box3d_pts_2d_pred, box3d_pts_3d_pred = utils.compute_box_3d(k, calib.P)
                iou_max.append(iou3d(box3d_pts_3d, box3d_pts_3d_pred)[0])
            if len(iou_max) ==0:
                return 1
            if max(iou_max) == 0:
                return 1

    return 0

def show_lidar_on_image(pc_velo, img, calib, img_width, img_height, side='left'):
    """ Project LiDAR points to image """
    if side == 'left':
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height, True)
    else:
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height, True, side='right')
    imgfov_pts_2d = pts_2d[fov_inds, :]
    print(imgfov_pts_2d.shape)

    # project those 3d pcl in velo coord to rect cam coord
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255


    img_test = np.zeros([img.shape[0], img.shape[1]])

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        print(img_test.shape)
        print("{}, {}".format(int(np.round(imgfov_pts_2d[i, 1])), int(np.round(imgfov_pts_2d[i, 0]))))
        img_test[int(np.round(imgfov_pts_2d[i, 1]))-1, int(np.round(imgfov_pts_2d[i, 0]))-1] = depth

    from PIL import Image
    from matplotlib import cm
    import matplotlib.pyplot as plt
    im = Image.fromarray(np.uint8(img_test))  # .convert('RGB')
    # im = Image.fromarray(np.uint8(cm.gist_earth(depth_map)))
    # im.save('/home/zelt/Desktop/books_read.png')
    plt.imshow(im, vmin=0, vmax=5)
    # plt.imsave('/home/zelt/Desktop/test/4_beams_right.png', im, vmin=0, vmax=30)  # , cmap='hot')
    plt.show()

    # print(imgfov_pts_2d.shape)



    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        # print(int(640.0 / depth))
        if int(640.0 / depth)>255:
            color = cmap[255, :]
        else:
            color = cmap[int(640.0 / depth), :]

        # print(tuple(color))

        cv2.circle( # draw each lidar pcl in fov by a circle
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            1,
            color=tuple(color),
            thickness=-1,
        )
    cv2.imshow("projection", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # cv2.imwrite('/home/zelt/Desktop/02.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def depth_map_from_lidar(pc_velo, img, calib, img_width, img_height, depth=False):
    """ Project LiDAR points to image """
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height, True, depth=depth)

    # pcl in image 2
    imgfov_pts_2d = pts_2d[fov_inds, :]
    # project those 3d pcl in velo coord to rect cam coord
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    # depth_map = np.zeros(img.shape[:2], dtype=np.uint16)
    depth_map = np.zeros(img.shape[:2])

    # imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    # imgfov_pts_2d = np.round(imgfov_pts_2d).astype('uint16')

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2] *256.0 # need to decode after
        depth_map[int(imgfov_pts_2d[i, 1])-1, int(imgfov_pts_2d[i, 0])-1] = depth
        # depth_map[int(np.round(imgfov_pts_2d[i, 1])), int(np.round(imgfov_pts_2d[i, 0]))] = depth

    depth_map = depth_map.astype('uint16')
    # cv2.imshow("projection", depth_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return depth_map



