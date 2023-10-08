import gym
import gym_unrealcv
from gym.spaces.box import Box
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor

def create_env(env_id, seed=1):
    env = gym.make(env_id)
    env = ResizeWrapper(env, 224)
    env.seed(seed)
    return env

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, size):
        super(ResizeWrapper, self).__init__(env)
        self.size = size
        self.observation_space = Box(low=0, high=255, shape=(3, size, size), dtype=np.uint8)  # image
    
    def observation(self, obs):
        """Return observation as opencv format, that is, in BGR and (H, W, C) 
        """        
        img = cv2.resize(obs, (self.size, self.size))
        return img
        
    def start(self):
        self.env.unwrapped.start()