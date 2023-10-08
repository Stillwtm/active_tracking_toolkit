import time
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward
from gym_unrealcv.envs.utils import env_unreal, misc
from gym_unrealcv.envs.tracking.interaction import Tracking

''' 
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
'''


class UnrealCvGeometry(gym.Env):
    def __init__(self,
                 setting_file,
                 category=0,
                 reset_type=0,
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(640, 480)
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0

        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.max_steps = setting['max_steps']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.reset_area = setting['reset_area']
        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.target_num = setting['target_num']
        self.exp_distance = setting['exp_distance']
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 30)

        self.obstacle_list = setting['obstacles']
        self.texture_interval = setting['texture_interval']
        self.texture_cnt = 0

        self.textures_list = misc.get_textures(setting['imgs_dir'], self.docker)
        # print(self.textures_list)

        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(docker, resolution)

        # connect UnrealCV
        self.unrealcv = Tracking(cam_id=self.cam_id, port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)

        # define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd'
        self.observation_space = self.unrealcv.define_observation(self.cam_id, self.observation_type, 'fast')

        # define reward type
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        self.rendering = False
        # self.unrealcv.start_walking(self.target_list[0]) snorlax
        self.count_steps = 0
        self.count_close = 0
        self.unrealcv.pitch = self.pitch

    def step(self, action=None):
        """
        action=None:
            store image, perform gt action, return [img, gt_action]
        action!=None:
            perform offered action, store image, return img
        """
        info = dict(
            Collision=False,
            Done=False,
            Trigger=0.0,
            Reward=0.0,
            Action=action,
            Pose=[],
            Trajectory=self.trajectory,
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
        )
        
        if action is not None:
            # perform offered action
            _, cur_yaw, cur_pitch = self.unrealcv.get_rotation(self.cam_id)
            # act_yaw, act_pitch = self.discrete_actions[action]
            # self.unrealcv.set_rotation(self.cam_id, [0, cur_yaw + act_yaw, cur_pitch + act_pitch])
            self.unrealcv.set_rotation(self.cam_id, [0, cur_yaw + action[0], cur_pitch + action[1]])
            # get image
            state = self.unrealcv.get_observation(self.cam_id, self.observation_type, 'fast')
            # state = [img, [0., 0.]]
        else:
            # get target location
            target_pos = self.unrealcv.get_obj_location(self.target_list[0])
            camera_pos = self.unrealcv.get_location(self.cam_id)
            _, cur_yaw, cur_pitch = self.unrealcv.get_rotation(self.cam_id)
            # abs camera rotation
            abs_yaw = 90 - np.arctan((target_pos[0]-camera_pos[0]) / (target_pos[1]-camera_pos[1])) / np.pi * 180
            xy = ((target_pos[0]-camera_pos[0])**2 + (target_pos[1]-camera_pos[1])**2)**0.5
            abs_pitch = np.arctan((target_pos[2] - camera_pos[2]) / xy) / np.pi * 180
            # perform abs action
            self.unrealcv.set_rotation(self.cam_id, [0, abs_yaw, abs_pitch])
            # calculate gt action
            gt_delta_yaw = abs_yaw - cur_yaw
            gt_delta_pitch = abs_pitch - cur_pitch
            # get image
            state = self.unrealcv.get_observation(self.cam_id, self.observation_type, 'fast')
            # state = [img, [gt_delta_yaw, gt_delta_pitch]]

        self.count_steps += 1                                         

        self.texture_cnt += 1
        if self.texture_cnt > self.texture_interval:
            # print("Texture Change!!!")
            self.texture_cnt = 0
            self.unrealcv.random_light(self.light_list)
            self.unrealcv.random_texture(self.background_list, self.textures_list)
            # self.unrealcv.random_obstacles(self.obstacle_list, self.textures_list,
            #                             10, self.reset_area, self.start_area, texture=True)
        
        # calculate reward
        info['Pose'] = self.unrealcv.get_pose(self.cam_id)
        self.target_pos = self.unrealcv.get_obj_pose(self.target_list[0])
        info['Direction'] = misc.get_direction(info['Pose'], self.target_pos)   # [-180, 180]
        direction_error = abs(info['Direction']) / (self.max_direction / 2)
        info['Reward'] = 1 - direction_error / (360 / self.max_direction)   # [0, 1]
        
        # if abs(info['Direction']) > self.max_direction / 2:
        #     # self.count_close += 1
        #     print("count_close: ", self.count_close)
        # else:
        #     self.count_close = 0

        if self.count_close >= 10:
            print("!!!!!!Lost!!!!!!!")
            info['Done'] = True
        
        if self.count_steps >= self.max_steps:
            info['Done'] = True

        return state, np.float(info['Reward']), info['Done'], info

    def reset(self):
        # self.C_reward = 0
        self.count_close = 0

        self.unrealcv.set_obj_location(self.target_list[0], self.safe_start[0])

        # set camera loaction and rotation, location
        cam_pos_exp = [0, -890, 700]
        self.unrealcv.set_location(self.cam_id, cam_pos_exp)
        time.sleep(0.5)

        # get target location
        target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        camera_pos = self.unrealcv.get_location(self.cam_id)
        # abs camera rotation
        abs_yaw = 90 - np.arctan((target_pos[0]-camera_pos[0]) / (target_pos[1]-camera_pos[1])) / np.pi * 180
        xy = ((target_pos[0]-camera_pos[0])**2 + (target_pos[1]-camera_pos[1])**2)**0.5
        abs_pitch = np.arctan((target_pos[2] - camera_pos[2]) / xy) / np.pi * 180
        # perform abs action
        self.unrealcv.set_rotation(self.cam_id, [0, abs_yaw, abs_pitch])

        # self.roll, yaw, self.pitch = 0, 90, -60
        # self.unrealcv.set_rotation(self.cam_id, [self.roll, yaw, self.pitch])
        # current_pose = self.unrealcv.get_pose(self.cam_id, 'soft')

        # get observation
        time.sleep(0.5)
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, 'fast')

        # save trajectory
        self.trajectory = []
        # self.trajectory.append(current_pose)
        self.count_steps = 0
        

        self.unrealcv.random_light(self.light_list)
        self.unrealcv.random_texture(self.background_list, self.textures_list)
        # self.unrealcv.random_obstacles(self.obstacle_list, self.textures_list,
        #                             10, self.reset_area, self.start_area, texture=True)
        
        # self.unrealcv.start_move(self.target_list[0])

        print("Reset done!")

        return state

    def start(self):
        self.unrealcv.start_move(self.target_list[0])

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color

    def seed(self, seed=None):
        pass

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area
