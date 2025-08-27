from sai_rl import SAIClient
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
import reward_fn as rw

def main():

    def make_env(env_id="FrankaIkGolfCourseEnv-v0"):
        def _init():
            sai = SAIClient("FrankaIkGolfCourseEnv-v0", api_key="put_api_key_here")
            env = gym.make(env_id, render_mode = None, max_episode_steps = 350)
            env = rw.FullGripRewardWrapperCustom(env)   
            return env
        return _init


    num_envs = 4  
    vec_env = SubprocVecEnv([make_env() for i in range(num_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    total_timesteps = 2_000_000  
    save_dir = "gripnet/"

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,        
        buffer_size=1_000_000,     
        learning_starts=10_000,    
        batch_size=512,           
        tau=0.005,                 
        gamma=0.99,                
        train_freq=(1, "step"),   
        gradient_steps=1,         
        ent_coef=0.005,      
        use_sde=True,            
        verbose=1,
        device="cuda"  
    )

    checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='gripnet_checkpoint_8/',
    name_prefix='sac_model',
    save_vecnormalize=True,  
    )

    model.learn(
        total_timesteps=total_timesteps, callback=checkpoint_callback,
        log_interval=50,
        progress_bar=True
    )

    final_path = os.path.join(save_dir, "sac_model_final.zip")
    final_path_vec = os.path.join(save_dir, "normalize.zip")
    model.save(final_path)
    vec_env.save(final_path_vec)
    print(f"Training finished. Final model saved to {final_path}")

if __name__ == "__main__":
    main()