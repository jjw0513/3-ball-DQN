import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np

# WandB API 사용
api = wandb.Api()
entity, project = "hails", "3ball_CAP"  # entity와 프로젝트 이름 설정
runs = api.runs(f"{entity}/{project}")

# 선택할 run 이름 리스트
selected_run_names = ["DQN", "A2C", "Dreamer", "TransDreamer"]

# 데이터 수집
data_list = []

for run in runs:
    if run.name in selected_run_names:
        data = {
            'run_id': run.id,
            'run_name': run.name,
            'config': run.config,
            'history': run.history(keys=['reward']),
        }
        data_list.append(data)

# 데이터프레임 생성
runs_df = pd.DataFrame(data_list)

# Time Weighted EMA 함수 정의 (스무딩 정도를 100으로 적용)
def time_weighted_ema(data, smoothing_degree=100):
    alpha = 1.0 / smoothing_degree  # smoothing_degree에 따라 alpha 계산
    ema = np.zeros_like(data, dtype=np.float64)
    ema[0] = data[0]  # 초기 값 설정

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]  # EMA 계산

    return ema

# 그래프 그리기
fig, ax = plt.subplots()

reward_arrays = []

# 각 run에 대한 리워드 데이터 처리
for index, row in runs_df.iterrows():
    rewards = row['history']['reward'].dropna().tolist()  # NaN 값 제거 후 리스트로 변환
    if len(rewards) > 0:
        # Time Weighted EMA 적용 (스무딩 정도 100)
        smoothed_rewards = time_weighted_ema(np.array(rewards), smoothing_degree=110)

        # x축은 원래 리워드의 길이에 맞춤
        x_target = np.arange(len(rewards))

        reward_arrays.append((x_target, smoothed_rewards))

# 각 run의 스무딩된 리워드 그리기 (선 스타일 반영)
styles = {
    "DQN": ("g-", "DQN"),
    "A2C": ("r-", "A2C"),
    "Dreamer": ("b-", "Dreamer"),
    "TransDreamer": ("m-", "TransDreamer")
}

for index, (x_target, smoothed_rewards) in enumerate(reward_arrays):
    line_style, label = styles[runs_df.iloc[index]['run_name']]
    ax.plot(x_target, smoothed_rewards, line_style, label=label)

# 그래프 설정
ax.set_xlabel('Episodes')
ax.set_ylabel('Reward')
ax.set_xlim([0, max(x_target)])  # x축 범위를 원래의 에피소드 길이에 맞춤
ax.set_xticks(np.linspace(0, max(x_target), 6))  # x축 눈금 설정
ax.set_xticklabels([0, 200, 400, 600, 800, 1000])  # x축 눈금 레이블 설정
ax.set_ylim(bottom=0)  # y축의 최솟값을 0으로 설정
ax.legend(loc='best')

plt.title('Reward with Time Weighted EMA')  # 제목 추가
plt.show()
