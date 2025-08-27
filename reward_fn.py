import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from scipy.spatial.transform import Rotation as R
from gymnasium import RewardWrapper
import os
import math

_FLOAT_EPS = np.finfo(np.float64).eps


def distance_between_objects(pos1, pos2):
    return abs(np.linalg.norm(pos1 - pos2))


def check_grasp(env):
    club_body_id = env.golf_club_id
    left_finger_body_id = env.left_finger_body_id
    right_finger_body_id = env.right_finger_body_id
    r = 0
    ncon = env.robot_model.data.ncon
    if ncon == 0:
        return False, r

    contact = env.robot_model.data.contact

    club_left_contact = False
    club_right_contact = False

    for i in range(ncon):
        geom1 = contact[i].geom1
        geom2 = contact[i].geom2

        body1 = env.robot_model.model.geom_bodyid[geom1]
        body2 = env.robot_model.model.geom_bodyid[geom2]

        if body1 == club_body_id or body2 == club_body_id:
            other_body = body2 if body1 == club_body_id else body1

            if other_body == left_finger_body_id:
                club_left_contact = True
            elif other_body == right_finger_body_id:
                club_right_contact = True

            if club_left_contact or club_right_contact:
                r = 0.5
            else:
                r = 0

    return (club_left_contact and club_right_contact), r

def ball_in_hole(env):

    ball_pos = env.robot_model.data.xpos[env.golf_ball_id]

    hole_pos = env.robot_model.data.xpos[env.golf_hole_id]

    return distance_between_objects(ball_pos, hole_pos) < 0.06


def ball_dis(env):
    ball_pos = env.robot_model.data.xpos[env.golf_ball_id]

    hole_pos = env.robot_model.data.xpos[env.golf_hole_id]

    return distance_between_objects(ball_pos, hole_pos) 

def check_ball_club_contact(env):
    club_body_id = env.club_head_id
    ball_body_id = env.golf_ball_id
    ncon = env.robot_model.data.ncon
    if ncon == 0:
        return False

    contact = env.robot_model.data.contact

    for i in range(ncon):
        geom1 = contact[i].geom1
        geom2 = contact[i].geom2

        body1 = env.robot_model.model.geom_bodyid[geom1]
        body2 = env.robot_model.model.geom_bodyid[geom2]

        if (body1 == club_body_id and body2 == ball_body_id) or (
            body1 == ball_body_id and body2 == club_body_id
        ):
            return True

    return False


def evaluation_fn(env):
    
    ee_pos = env.robot_model.data.site(env.ee_site_id).xpos
    club_grip_pos = env.robot_model.data.xpos[env.golf_club_id]
    club_head_pos = env.robot_model.data.xpos[env.club_head_id]
    ball_pos = env.robot_model.data.xpos[env.golf_ball_id]

    ee_club_dist = distance_between_objects(ee_pos, club_grip_pos)
    club_ball_dist = distance_between_objects(club_head_pos, ball_pos)
    hole_in_one = ball_dis(env)

    return math.exp(-5*ee_club_dist) , math.exp(-5*club_ball_dist) , math.exp(-5*hole_in_one) 
  
class FullGripRewardWrapperCustom(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.timer = 0
        self.e = False
        self.b = True
        self.h = True
        self.i = 0
        self.j = 0
        self.old_e = 0
        self.old_b = 0
        self.old_h = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 0

        e, b, h = evaluation_fn(self.env.unwrapped)

        
        progress_e = e - self.old_e
        progress_b = b - self.old_b
        progress_h = h - self.old_h


        reward = progress_e*(1 - int(self.e)) + progress_b*(1-int(self.b)) + progress_h*(1-int(self.h))

        self.old_e = e
        self.old_b = b
        self.old_h = h

        contact, bonus = check_grasp(self.env.unwrapped)

        if not self.e and self.j < 2 and bonus !=0:
            reward += bonus
            self.j += 1
    

        if contact and not self.e:
            reward += max(1 - self.timer*0.001, 0.3)
            if self.i <= 2:
                self.e = True
                self.b = False
            self.i+= 1
        if check_ball_club_contact(self.env.unwrapped) and self.e and not self.b:
            reward += max(1.67 - self.timer*0.001, 0.5)
            self.b = True
            self.h = False
        if ball_in_hole(self.env.unwrapped):
            reward += 10
            self.flag = True
            self.h = True
            self.e = False
            self.timer = 0
            self.i = 0
            self.j = 0
            self.old_e = 0
            self.old_b = 0
            self.old_h = 0
            self.truncated = True
        if terminated:
            reward -= 0.1
            self.timer = 0
            self.old_e = 0
            self.old_b = 0
            self.old_h = 0
            self.h = True
            self.b = True
            self.e = False
            self.i = 0
            self.j = 0
        if truncated:
            self.old_e = 0
            self.old_b = 0
            self.old_h = 0
            self.timer = 0
            self.h = True
            self.b = True
            self.e = False
            self.i = 0
            self.j = 0
            

        self.timer += 1
        return obs, reward, terminated, truncated, info