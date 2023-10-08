import numpy as np
import cv2

class CameraBase(object):
    def __init__(self):
        pass
    
    def move(self, img, bbox):
        """generate camera movement

        Args:
            img (numpy array): CxHxW
            bbox (numpy array): (4,), [cx, cy, w, h]

        Returns:
            (array): (2,), delta (yaw, pitch)
        """ 
        raise NotImplementedError

class CameraDiscrete(CameraBase):
    """a camera that perform fixed simple discrete action
    """
    def __init__(self, center_ratio=0.25):
        super().__init__()
        self.center_ratio = center_ratio
        # no-op, up, down, left, right, in form of delta (yaw, pitch) 
        self.discrete_action = [[0, 0], [0, 10], [0, -10], [-10, 0], [10, 0]]
        # self.discrete_action = [[0, 0], [0, 5], [0, -5], [-5, 0], [5, 0]]

    def move(self, img, bbox):
        h, w, _ = img.shape
        r = self.center_ratio * min(w, h)
        # treat center of image as origin and invert y-axis
        cx = bbox[0] + bbox[2] / 2 - w / 2
        cy = -(bbox[1] + bbox[3] / 2 - h / 2)
        d = np.sqrt(cx * cx + cy * cy)
        k = h / w
        # print(f"cx:{cx}, cy:{cy}, d:{d}, k:{k}, h:{h}, w:{w}")
        act_id = 0
        if d < r:
            act_id = 0
        elif cy > abs(k * cx):  # up
            act_id = 1
        elif cy < -abs(k * cx):  # down
            act_id = 2
        elif abs(cy) < -k * cx:  # left
            act_id = 3
        elif abs(cy) < k * cx:  # right
            act_id = 4

        return self.discrete_action[act_id]

class CameraOpt(CameraBase):
    """A camera that can accurately point towards a specified direction
    """
    def __init__(self, fov):
        super().__init__()
        self.fov = fov
    
    def move(self, img, bbox, distance):
        #  focal length = image_width / 2 / tan(FOV/2)
        h, w, _ = img.shape
        fx = fy = w / 2 / np.tan(self.fov / 2)
        cv2.point
    
    def _depth_conversion(point_depth, f):
        """convert depth to camera center to depth to camera plane
        """
        h, w = point_depth.shape
        i_c = np.float(h) / 2 - 1
        j_c = np.float(w) / 2 - 1
        cols, rows = np.meshgrid(np.linspace(0, w - 1, num=w), np.linspace(0, h - 1, num=h))
        dist_c = ((rows - i_c)**2 + (cols - j_c)**2)**0.5
        plane_depth = point_depth / (1 + (dist_c / f)**2)**0.5
        return plane_depth
        
        
    
        