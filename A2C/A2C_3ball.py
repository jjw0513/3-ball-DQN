
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from envs.GymMoreRedBalls import GymMoreRedBalls
from envs.wrapper import GymMoreRedBallsWrapper, FullyCustom, MaxStepsWrapper, ActionSpaceWrapper
import wandb
from wandb.integration.sb3 import WandbCallback

# Initialize WandB
run = wandb.init(project="3ball_CAP", entity='hails', sync_tensorboard=True, config={
    "learning_rate": 2e-4,  # Dreamer의 model_lr
    "n_steps": 50,  # Dreamer의 batch_length
    "gamma": 0.999,  # Dreamer의 discount
    "gae_lambda": 0.95,  # Dreamer의 lambda_
    "ent_coef": 0.0,  # 기본 설정
    "vf_coef": 0.5,  # 기본 설정
    "max_grad_norm": 100.0,  # Dreamer의 grad_clip
    "rms_prop_eps": 1e-5,  # Dreamer의 eps
    "normalize_advantage": False,
    "max_episodes": 1000,
    "max_steps": 5000,
})

# Environment creation function
def make_env():
    env = GymMoreRedBalls(room_size=10, render_mode="rgb_array")
    env = GymMoreRedBallsWrapper(env, max_steps=5000)
    env = ActionSpaceWrapper(env, max_steps=5000, new_action_space=3)
    env = FullyCustom(env, max_steps=5000)
    return MaxStepsWrapper(env, max_steps=5000)

# Create a vectorized environment
env = DummyVecEnv([make_env])

# Create the A2C model
model = A2C(
    "MlpPolicy",
    env,
    learning_rate=wandb.config.learning_rate,
    n_steps=wandb.config.n_steps,
    gamma=wandb.config.gamma,
    gae_lambda=wandb.config.gae_lambda,
    ent_coef=wandb.config.ent_coef,
    vf_coef=wandb.config.vf_coef,
    max_grad_norm=wandb.config.max_grad_norm,
    rms_prop_eps=wandb.config.rms_prop_eps,
    verbose=1,
    tensorboard_log=f"runs/{run.id}"
)


model.learn()
