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


def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32)
    # print(obj.shape)
    try:
        obj = obj.reshape(-1, 4)
    except ValueError:
        obj = obj.reshape(-1, 5)
    # print(obj.shape)

    return obj


def load_depth_npy(depth_path):
    return np.load(depth_path)

def load_image(img_filename):
    img = cv2.imread(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # left
    return img

def transform_original_image(img):
    # Crop-center original image
    cropx, cropy = 1216, 352
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def load_image1(img_filename):
    try:
        from PIL import Image
    except ImportError:
        import Image

    im = Image.open(img_filename)
    width, height = im.size   # Get dimensions
    new_width = 1216
    new_height = 352
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return np.array(im)


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

    def __init__(self, velo_to_cam, cam_to_cam, from_video=False):
        if from_video:
            pass
            # calibs = self.read_calib_from_video(velo_to_cam, cam_to_cam)
        else:
            calibs = self.read_calib_file(velo_to_cam, cam_to_cam)
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

    def read_calib_file(self, velo_to_cam, cam_to_cam):
        """
        Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        :param filepath:
        :return:
        """
        data = {}
        data2 = {}

        data_new = {}

        
        with open(cam_to_cam, 'r') as f:
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
        
        with open(velo_to_cam, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data2[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        """
        data3 = {}
        with open(imu_to_velo, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data3[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        """
        insert = np.insert(data2['R'], [3], data2['T'][0]) 
        insert = np.insert(insert, [7], data2['T'][1]) 
        Tr_velo_to_cam = np.insert(insert, [11], data2['T'][2]) 
        # insert = np.insert(data3['R'], [3], data3['T'][0]) 
        # insert = np.insert(insert, [7], data3['T'][1]) 
        # Tr_imu_to_velo = np.insert(insert, [11], data3['T'][2]) 
        # data_new['Tr_imu_to_velo'] = Tr_imu_to_velo
        data_new['Tr_velo_to_cam'] = Tr_velo_to_cam
        data_new['P0'] = data['P_rect_00']
        data_new['P1'] = data['P_rect_01']
        data_new['P2'] = data['P_rect_02']
        data_new['P3'] = data['P_rect_03']
        data_new['R0_rect'] = data['R_rect_00']
        return data_new
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


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

# https://github.com/windowsub0406/KITTI_Tutorial/blob/master/velo2cam_projection.ipynb
# https://programtalk.com/vs2/python/13601/pykitti/pykitti/raw.py/



class Calibration0(object):
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
