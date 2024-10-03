import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from envs.GymMoreRedBalls import GymMoreRedBalls
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

# WandB 초기화
run = wandb.init(
    project="3ball_CAP",
    entity='hails',
    sync_tensorboard=True,
    config={
        "learning_rate": 2e-4,  # Dreamer의 model_lr
        "n_steps": 50,  # Dreamer의 batch_length
        "gamma": 0.999,  # Dreamer의 discount
        "gae_lambda": 0.95,  # Dreamer의 lambda_
        "ent_coef": 0.0,  # 기본 설정
        "vf_coef": 0.5,  # 기본 설정
        "max_grad_norm": 100.0,  # Dreamer의 grad_clip
        "rms_prop_eps": 1e-5,  # Dreamer의 eps
        "normalize_advantage": False,
        "max_episodes": 1000,  # 최대 에피소드 수
        "max_steps": 5000,  # 최대 스텝 수
    }
)


# 환경 생성 함수
def make_env():
    env = GymMoreRedBalls(room_size=10, render_mode="rgb_array")
    # Wrappers는 필요한 경우 추가할 수 있습니다.
    return env


# 사용자 정의 A2C 클래스
class CustomA2C(A2C):
    def __init__(self, *args, **kwargs):
        super(CustomA2C, self).__init__(*args, **kwargs)
        self.schedule = 'linear'  # 기본 schedule 값을 'linear'로 설정

    def learn(self, total_episodes=None, total_timesteps=None, **kwargs):
        episodes_so_far = 0
        timesteps_so_far = 0

        while True:  # 학습 루프
            # 모델 학습 코드
            result = super().learn(total_timesteps=total_timesteps, **kwargs)
            timesteps_so_far += result

            # 에피소드 수 증가
            episodes_so_far += 1

            # 에피소드 수 기반 종료 조건
            if total_episodes is not None and episodes_so_far >= total_episodes:
                break


# 벡터화된 환경 생성
env = DummyVecEnv([make_env])

# 모델 생성
model = CustomA2C(
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

# 모델 학습
model.learn(total_episodes=wandb.config.max_episodes, callback=WandbCallback())

# WandB 종료
wandb.finish()
