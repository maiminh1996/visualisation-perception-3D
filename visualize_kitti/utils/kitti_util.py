import os

import cv2
import numpy as np
import sys

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
sys.path.insert(0, MY_DIRNAME)

from math_util import *

# np.random.seed(0)


class Object3d(object):
    """
    3d object label
    """

    def __init__(self, label_file_line):
        """
        type:       'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                    'Cyclist', 'Tram', 'Misc' or 'DontCare'
        truncated:  Float, 0: non-truncated or 1: truncated
                    truncated which refers to the object leaving image boundaries
        occluded:   Integer (0, 1, 2, 3) indicating occlusion state:
                    0 = fully visible, 1 = partly occluded
                    2 = largely occluded, 3 = unknown
        alpha:      Observation angle of obj [-pi;pi]
        bbox:       2D bbox of object in the image (0-based camera index):
                    (left, top, right, bottom in pixel coord)
        dimensions: 3D obj dims: height, width, length (in meters)
        location:   3D obj location (x,y,z) in cam coord (in meters)
        rotation_y: Rotation ry around Y-axis in cam coord [-pi;pi]
        Ref:        README.txt in  devkit_object.zip (object development kit)
                    downloaded from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
        :param label_file_line: a line in label file
        """
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bbox in 0-based coord (left gray cam)
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bbox information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in came coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        """
        Interprate each object in label_2 file
        :return: nothing
        """
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


class Calibration(object):
    """
    Calibration matrices and utils
    3d XYZ in <label>.txt are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in Velodyne coord.

    y_image2 = P^2_rect * x_rect
    y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
    x_ref = Tr_velo_to_cam * x_velo
    x_rect = R0_rect * x_ref

    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                0,      0,      1,      0]
             = K * [1|t]

    image2 coord:
     ----> x-axis (u)
    |
    |
    v y-axis (v)

    velodyne coord:
    front x, left y, up z

    rect/ref camera coord:
    right x, down y, front z

    Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

    TODO(rqi): do matrix multiplication only once for each projection.

    """

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        self.P_3 = calibs['P3']
        self.P_3 = np.reshape(self.P_3, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        """
        Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        :param filepath:
        :return:
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        """
        X (n,3) :--> (X|1) (n, 4)
        :param pts_3d: nx3 points in Cartesian
        :return: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        """
        project a 3D point from velo coord into ref cam coord
        :param pts_3d_velo: 3d point in velo coord (n, 3)
        :return: (n, 3) points into ref cam coord
        V_ref = (V_velo) * (V2C).T
        V2C == Tr_velo_to_cam (3x4)
        """
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        """
        project a 3D point from ref cam coord into velo coord
        :param pts_3d_ref:
        :return:
        V_rect = V_ref * (C2V).T
        C2V == Tr_cam_to_velo (3x4)
        """
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        """
        Project a point 3D in rectified cam coord into reference cam coord (left gray cam)
        Input and Output are nx3 points
        :param pts_3d_rect:
        :return:
        V_ref = ((R0)^-1 * (V_rect).T).T
        R0 (3x3)
        """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """
        project a 3D point from ref cam coord into rect cam coord
        :param pts_3d_velo: 3d point in ref cam coord (n, 3)
        :return: (n, 3) points into rect cam coord
        V_rect = (R0) * (V_ref).T
        R0 (3x3)
        """
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """
        project a 3D point from rect cam coord into velo coord
        :param pts_3d_velo: 3d point in rect cam coord (n, 3)
        :return: (n, 3) points into velo coord
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        """
        project a 3D point from velo coord into rect cam coord
        :param pts_3d_velo: 3d point in velo coord (n, 3)
        :return: (n, 3) points into rect cam coord
        velo --> ref --> rect
        """
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect, depth=False, side='left'):
        """
        project a 3D point from rect cam coord into image (left color cam)
        :param pts_3d_rect: 3d point in rect cam coord (n, 3)
        :return:
        V_img = ((V_rect.T)|1) * P2.T
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        if side=='left':
            pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        else:
            pts_2d = np.dot(pts_3d_rect, np.transpose(self.P_3))  # nx3
        # scale projected point center in image plane to top left corner in image
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        if depth==True:
            return pts_2d
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo, depth=False, side='left'):
        """
        project a 3D point from velo coord into image coord
        :param pts_3d_velo: 3d point in velo coord (n, 3)
        :return: (n, 2) points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect, depth=depth) if side=='left' else self.project_rect_to_image(pts_3d_rect, depth=depth, side=side)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


# =========================================
# ------- for calibration object ----------
# =========================================

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


# =========================================
# -----------------------------------------
# =========================================

# ===================================
# ------- for kitti object ----------
# ===================================

def read_label(label_filename):
    """

    :param label_filename: .../000001.txt ex
    :return:
    """
    lines = [line.rstrip() for line in open(label_filename)]  # list all objects in an image
    objects = [Object3d(line) for line in lines]  # process each object in image
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_velo_scan(velo_filename):
    """
    Load lidar pcl from .bin file
    :param velo_filename: .bin file
    :return: (n, 4), n point cloud, 4: x, y, z, reflectance values
    """
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # scan = scan.reshape((-1, 5))[:, 0:4] # fog ori
    return scan


# ===================================
# -----------------------------------
# ===================================

def compute_box_3d(obj, P, rx=0, rz=0):
    """
    Takes an object and a projection matrix (P) and projects the 3d
    bounding box into the image plane.
    :param obj:
    :param P:
    :return: corners_2d: (8,2) array in left image coord.
             corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    # in cam coord yaw axis --> y axis
    # R = np.dot(rotz(rz), np.dot(roty(obj.ry), rotx(rx)))
    # t = np.array(obj.t).reshape([len(obj.t), 1])
    # rotate and translate 3d bounding box [R|t] (3, 4)
    # Rt_matrix = np.hstack((R, t))
    Rt_matrix = rotation_translation_matrix(rx, obj.ry, rz, obj.t)
    # 3d bounding box dimensions of object
    l = obj.l  # length dai
    w = obj.w  # width rong
    h = obj.h  # height cao
    # 3d bbox corners in object coord sys with x,y,z coord following rect cam coord
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    eight_corners = np.vstack([x_corners, y_corners, z_corners, np.ones([len(x_corners)])])  # (4, 8)
    corners_3d = np.dot(Rt_matrix, eight_corners)

    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):  # <0.1m
        print("-------------error right here, ", corners_3d[2, :])  # TODO try
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    return corners_2d, np.transpose(corners_3d)


def project_to_image(pts_3d, P):
    """
    Project 3d points to image plane.
    :param pts_3d: nx3 matrix
    :param P: 3x4 projection matrix P2 for left color camera
    P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    :return: pts_2d: nx2 matrix
    """
    pts_3d_extend = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # (pts_3d|1) (n,4)
    pts_2d = np.dot(P, pts_3d_extend.T)  # (3,n)

    # respect from the center of the image plane to the origin of image which is at the top left corner.
    # (u', v', w') --> (u, v) = (u'/w', v'/w')
    # pts_2d = pts_2d.T  # (n,3)
    # pts_2d[:, 0] /= pts_2d[:, 2]
    # pts_2d[:, 1] /= pts_2d[:, 2]
    # return pts_2d[:, 0:2]
    pts_2d_uv = (pts_2d[0:2, :] / pts_2d[2, :]).T
    return pts_2d_uv


def compute_orientation_3d(obj, P, rx=0, rz=0):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    # R = roty(obj.ry)

    # orientation in object coordinate system
    # TODO to understand how to represent orientation in obj coord sys
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    # orientation_3d = np.dot(R, orientation_3d)
    #  orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    # orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    # orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]
    orientation_3d = np.vstack([orientation_3d, np.ones([orientation_3d.shape[-1]])])  # (4, 2)
    Rt_matrix = rotation_translation_matrix(rx, obj.ry, rz, obj.t)
    orientation_3d = np.dot(Rt_matrix, orientation_3d)

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P);
    return orientation_2d, np.transpose(orientation_3d)


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    """
    Draw 3d bounding box in image
    :param image:
    :param qs: (8,2) array of vertices for the 3d box in following order
               8 projected points
    :param color:
    :param thickness:
    :return:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv 3   # cv2.CV_AA for opencv 2.4.x
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image


import math

TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 100
TOP_Z_MIN = -3.5
TOP_Z_MAX = 0.6

TOP_X_DIVISION = 0.2
TOP_Y_DIVISION = 0.2
TOP_Z_DIVISION = 0.3


def lidar_to_top(lidar):
    idx = np.where(lidar[:, 0] > TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 0] < TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 1] > TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 1] < TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 2] > TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 2] < TOP_Z_MAX)
    lidar = lidar[idx]

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]
    qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
    qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
    # qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION
    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    Z0, Zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
    height = Xn - X0
    width = Yn - Y0
    channel = Zn - Z0 + 2
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height, width, channel), dtype=np.float32)

    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  # new method
        for x in range(Xn):
            ix = np.where(quantized[:, 0] == x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0:
                continue
            yy = -x

            for y in range(Yn):
                iy = np.where(quantized_x[:, 1] == y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if count == 0:
                    continue
                xx = -y

                top[yy, xx, Zn + 1] = min(1, np.log(count + 1) / math.log(32))
                max_height_point = np.argmax(quantized_xy[:, 2])
                top[yy, xx, Zn] = quantized_xy[max_height_point, 3]

                for z in range(Zn):
                    iz = np.where(
                        (quantized_xy[:, 2] >= z) & (quantized_xy[:, 2] <= z + 1)
                    )
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0:
                        continue
                    zz = z

                    # height per slice
                    max_height = max(0, np.max(quantized_xyz[:, 2]) - z)
                    top[yy, xx, zz] = max_height

    # if 0: #unprocess
    #     top_image = np.zeros((height,width,3),dtype=np.float32)
    #
    #     num = len(lidar)
    #     for n in range(num):
    #         x,y = qxs[n],qys[n]
    #         if x>=0 and x <width and y>0 and y<height:
    #             top_image[y,x,:] += 1
    #
    #     max_value=np.max(np.log(top_image+0.001))
    #     top_image = top_image/max_value *255
    #     top_image=top_image.astype(dtype=np.uint8)

    return top


def draw_top_image(lidar_top):
    top_image = np.sum(lidar_top, axis=2)
    top_image = top_image - np.min(top_image)
    divisor = np.max(top_image) - np.min(top_image)
    top_image = top_image / divisor * 255
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image


def draw_box3d_on_top(
        image,
        boxes3d,
        color=(255, 255, 255),
        thickness=1,
        scores=None,
        text_lables=[],
        is_gt=False,
):
    # if scores is not None and scores.shape[0] >0:
    # print(scores.shape)
    # scores=scores[:,0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    num = len(boxes3d)
    startx = 5
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)
        if is_gt:
            color = (0, 255, 0)
            startx = 5
        else:
            color = heat_map_rgb(0.0, 1.0, scores[n]) if scores is not None else 255
            startx = 85
        cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    return img


def lidar_to_top_coords(x, y, z=None):
    if 0:
        return x, y
    else:
        # print("TOP_X_MAX-TOP_X_MIN:",TOP_X_MAX,TOP_X_MIN)
        X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
        Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
        xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
        yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)

        return xx, yy
