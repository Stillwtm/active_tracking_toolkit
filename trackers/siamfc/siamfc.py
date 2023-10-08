import torch
import numpy as np
import cv2
import time
from  tracker import TrackerBase
from .net import NetSiamFC
from .transforms import crop_resize_box, crop_resize_center, to_tensor
from .config import cfg


def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch

class TrackerSiamFC(TrackerBase):
    def __init__(self, model_path):
        super(TrackerSiamFC, self).__init__()

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.net = NetSiamFC(score_scale=1.0)  # 推理时不需要再缩放
        model_dict = torch.load(model_path)
        if 'model' in model_dict:
            model_dict = model_dict['model']
        self.net.load_state_dict(model_dict)
        self.net = self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def init(self, img, bbox):
        # 二维余弦窗口
        self.cos_window = np.outer(
            np.hanning(cfg.upscale_size),
            np.hanning(cfg.upscale_size)
        )
        self.cos_window /= self.cos_window.sum()
        # 搜索尺寸
        scale_exps = np.arange(cfg.scale_num) - (cfg.scale_num - 1) / 2
        self.scale_factors = cfg.scale_step**scale_exps

        self.center = bbox[:2] + bbox[2:] / 2
        self.target_size = bbox[2:]

        z, self.z_size = crop_resize_box(img, bbox, cfg.context_amount, 
            cfg.exemplar_size, cfg.exemplar_size)
        
        self.x_size = self.z_size / cfg.exemplar_size * cfg.instance_size
        self.kernel = self.net.backbone(
            to_tensor(z).to(self.device).unsqueeze(0))
        self.kernel = self.kernel.repeat(cfg.scale_num, 1, 1, 1)

    @torch.no_grad()
    def track(self, img):
        # 多个尺寸的搜索图像合并成一个tensor
        xs = [crop_resize_center(
            img, self.center, 
            (self.x_size * s, self.x_size * s),
            (cfg.instance_size, cfg.instance_size))
            for s in self.scale_factors]

        xs = torch.FloatTensor(
            np.array(xs)).permute(0, 3, 1, 2).to(self.device)

        xs = self.net.backbone(xs)
        scores = self.net.head(self.kernel, xs)
        scores = scores.squeeze(1).cpu().numpy()

        # 三次插值upscale，加上缩放scale的惩罚
        scores_up = [cv2.resize(score, (cfg.upscale_size, cfg.upscale_size), 
            interpolation=cv2.INTER_CUBIC) * cfg.scale_penalty for score in scores]
        scores_up[cfg.scale_num // 2] /= cfg.scale_penalty  # 未缩放的没有惩罚

        # 归一化后混合余弦窗口惩罚
        scale_id = np.argmax(np.amax(scores_up, axis=(1, 2)))
        scores_up = scores_up[scale_id]
        scores_up -= scores_up.min()
        scores_up /= scores_up.sum() + 1e-12
        scores_up = (1 - cfg.window_influence) * scores_up + \
            cfg.window_influence * self.cos_window

        # 将最大值对应坐标转换到原图中
        xy = np.unravel_index(np.argmax(scores_up), scores_up.shape)
        xy_in_scoreup = np.array(xy) - (cfg.upscale_size - 1) / 2  # 原点转换到中央
        xy_in_score = xy_in_scoreup / cfg.upscale_size * cfg.score_size
        xy_in_instance = xy_in_score * cfg.stride
        xy_in_img = xy_in_instance / cfg.instance_size * self.x_size * \
            self.scale_factors[scale_id]
        self.center += xy_in_img[::-1]  # 注意,img的shape是(h, w)

        # 更新scale
        scale = (1 - cfg.scale_lr) * 1. + cfg.scale_lr * \
            self.scale_factors[scale_id]
        self.z_size *= scale
        self.x_size *= scale
        self.target_size *= scale

        bbox = np.array([
            self.center[0] - (self.target_size[0] - 1) / 2,
            self.center[1] - (self.target_size[1] - 1) / 2,
            self.target_size[0], self.target_size[1]
        ])

        return bbox
