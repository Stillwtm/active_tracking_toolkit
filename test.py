import torch
import numpy as np
import argparse
from env import create_env
from camera import CameraDiscrete
from trackers import TrackerSiamFC
from trackers import TrackerTransT
from utils import show_img, make_video

def test(env, tracker, camera, args):
    ar = 0  # AR, accumulated reward
    el = 0  # EL, episode length
    _, w, h = env.observation_space.shape
    init_bbox = np.array([-18. + w / 2, -18. + h / 2, 36., 36.])
    imgs = []
    
    try:
        state = env.reset()
        tracker.init(state, init_bbox)
        show_img(state, init_bbox, "init")
        env.start()

        for steps in range(args.max_test_steps):
            bbox = tracker.track(state)
            action = camera.move(state, bbox)
            img = show_img(state, bbox, "step")
            imgs.append(img)

        # self.unrealcv.start_move(self.target_list[0])
            state, reward, done, _ = env.step(action)
            ar += reward

            if done:
                el = steps + 1
                break
    finally:
        env.close()
        make_video(imgs, './test.mp4')

    return ar, el


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='UnrealTrack-GeometryTrackRam-DiscreteColor-v0', help='environment name to use')
    parser.add_argument('--max_test_steps', type=int, default=500, help='max steps for test')
    parser.add_argument('--test_num', type=int, default=int(1), help='test how many times')
    parser.add_argument('--model_path', type=str, help='pretrained model path')
    args = parser.parse_args()

    ars = []
    els = []
    for i in range(args.test_num):
        print(f"Test {i} is running......")
        env = create_env(args.env)
        tracker = TrackerSiamFC(args.model_path)
        # tracker = TrackerTransT(args.model_path)
        camera = CameraDiscrete(center_ratio=0.2)
        ar, el = test(env, tracker, camera, args)
        ars.append(ar)
        els.append(el)
        print(f"Test {i} finished.")

    print(f"AR;\tEL")
    for i in range(args.test_num):
        print(f"{ars[i]};\t{els[i]}")
    
