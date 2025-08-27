from sai_rl import SAIClient
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import reward_fn as rw


def make_env():
    sai = SAIClient("FrankaIkGolfCourseEnv-v0", api_key="put_api_key_here")
    env = gym.make("FrankaIkGolfCourseEnv-v0", render_mode = None, max_episode_steps = 350)
    env = rw.FullGripRewardWrapperCustom(env)   
 
    return env

model = SAC.load("path_to_model")
vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("path_to_normalization_variables", vec_env)
vec_env.training = False
vec_env.norm_reward =  False
obs = vec_env.reset()
rewards = []
done = False
episodes = 100
tot_rw = []
for i in range(episodes):
    obs = vec_env.reset()
    done = False
    rewards = []
    contact = False
    while not done:
        action, _states = model.predict(obs, deterministic = False)
        obs, reward, terminated, info = vec_env.step(action)
        rewards.append(reward)

    print(sum(rewards), len(rewards))
    tot_rw.append(sum(rewards))

print(sum(tot_rw)/len(tot_rw))
