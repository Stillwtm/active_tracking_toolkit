import cv2
import torch
import numpy as np

def crop_resize_box(img, box, context, exemplar_sz, out_sz, max_translate=0):
    """以box为中心进行crop并resize,用平均值填充

    Args:
        box: [xmin, ymin, w, h]
        out_sz: scalar, 代表正方形边长
    Returns:
        new_img: 裁切缩放后的图片
        z_sz: exemplar在原图中对应的大小 
    """
    cx, cy = box[:2] + box[2:] / 2
    if max_translate > 0:
        cx += np.random.randint(-max_translate, max_translate+1)
        cy += np.random.randint(-max_translate, max_translate+1)
    w, h = box[2:]
    nw, nh = context * (w + h) + box[2:]
    z_sz = np.sqrt(nw * nh)
    n_sz = z_sz * out_sz / exemplar_sz
    new_img = crop_resize_center(img, (cx, cy), 
        (n_sz, n_sz), (out_sz, out_sz))
    return new_img, z_sz

def crop_resize_center(img, center, size, out_size):
    """给定中心进行crop并resize,用平均值填充

    Args:
        center: (x, y)
        size: (w, h)
        out_size: (w, h)
    Returns:
        new_img: 裁切缩放后的图片
    """
    cx, cy = center
    w, h = size
    img_h = img.shape[0]
    img_w = img.shape[1]
    # padding
    pad = int(max(-1, (w - 1) / 2 - cx, (w - 1) / 2 + cx - img_w,
        (h - 1) / 2 - cy, (h - 1) / 2 + cy - img_h)) + 1
    if pad > 0:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, 
            cv2.BORDER_CONSTANT, value=img.mean(axis=(0, 1)))
    # crop
    if not isinstance(size, np.ndarray):
        size = np.array(size)
    tl = (center - (size - 1) / 2 + pad).astype(int)
    br = (center - (size - 1) / 2 + pad + size).astype(int)
    new_img = img[tl[1]:br[1], tl[0]:br[0]]  # 注意cv2的wh是反的
    # resize
    new_img = cv2.resize(new_img, out_size)
    return new_img

def to_tensor(img):
    return torch.FloatTensor(img).permute(2, 0, 1)

class TransformsSiamFC(object):
    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

    def __call__(self, z, x, box_z, box_x):
        z = self._random_stretch(z)
        x = self._random_stretch(x)
        z, _ = crop_resize_box(z, box_z, self.context, 
            self.exemplar_sz, self.exemplar_sz, 4)
        x, _ = crop_resize_box(x, box_x, self.context, 
            self.exemplar_sz, self.instance_sz - 2 * 8, 4)  # 这里-2*8也许可以减少一些负样本
        z = to_tensor(z)
        x = to_tensor(x)
        return z, x

    def _random_stretch(self, img, max_stretch=0.05):
        scale = 1. + np.random.uniform(-max_stretch, max_stretch)
        out_size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
        return cv2.resize(img, out_size)

    
