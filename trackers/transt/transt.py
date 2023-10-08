import numpy as np

from tracker import TrackerBase
from .net_wrappers import NetWithBackbone
from .tracker import TransT

class TrackerTransT(TrackerBase):
    def __init__(self, net_path):
        net = NetWithBackbone(net_path=net_path, use_gpu=True)
        self.tracker = TransT(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)
    
    def init(self, img, bbox):
        self.tracker.initialize(img, {'init_bbox': bbox})

    def track(self, img):
        res = self.tracker.track(img, {})
        return np.array(res['target_bbox'])