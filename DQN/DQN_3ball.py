import gymnasium as gym
import torch as th
import numpy as np
import random
import math
import wandb
from envs.GymMoreRedBalls import GymMoreRedBalls

batch_size = 64
lr = 0.0001
EPS_START = 1
EPS_END = 0.2
EPS_DECAY = 200
steps_done = 0
replay_memory_size = 10000
gamma = 0.9
target_update_iter = 200

# Initialize wandb and log hyperparameters
wandb.init(project="DQN_redball_3try", entity='hails', config={
    "batch_size": batch_size,
    "learning_rate": lr,
    "epsilon_start": EPS_START,
    "epsilon_end": EPS_END,
    "epsilon_decay": EPS_DECAY,
    "replay_memory_size": replay_memory_size,
    "gamma": gamma,
    "target_update_iter": target_update_iter,
})

#env = gym.make('MiniGrid-Empty-8x8-v0', render_mode='rgb_array')
env = GymMoreRedBalls(room_size=10, render_mode='rgb_array')
env = env.unwrapped

device = th.device("cuda" if th.cuda.is_available() else "cpu")
n_action = 3

if isinstance(env.observation_space, gym.spaces.Dict):
    n_state = np.prod(env.observation_space['image'].shape)
else:
    n_state = np.prod(env.observation_space.shape)

hidden = 32

class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = th.nn.Linear(n_state, hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = th.nn.Linear(hidden, n_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = th.nn.functional.relu(x)
        out = self.out(x)
        return out

class ReplayMemory:
    def __init__(self):
        self.memory_size = replay_memory_size
        self.memory = []
        self.cur = 0

    def size(self):
        return len(self.memory)

    def store_transition(self, trans):
        if len(self.memory) < self.memory_size:
            self.memory.append(trans)
        else:
            self.memory[self.cur] = trans
            self.cur = (self.cur + 1) % self.memory_size

    def sample(self):
        if len(self.memory) < batch_size:
            return -1
        sam = np.random.choice(len(self.memory), batch_size)
        batch = [self.memory[i] for i in sam]
        return np.array(batch, dtype=object)

class DQN:
    def __init__(self):
        self.eval_q_net, self.target_q_net = Net().to(device), Net().to(device)
        self.replay_mem = ReplayMemory()
        self.iter_num = 0
        self.optimizer = th.optim.Adam(self.eval_q_net.parameters(), lr=lr)
        self.loss_fn = th.nn.MSELoss().to(device)
        self.loss_history = []

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with th.no_grad():
                state = state.unsqueeze(0)  # 2차원 텐서로 변환
                return self.eval_q_net(state.to('cpu')).max(1)[1].view(1, 1)
        else:
            return th.tensor([[random.randrange(n_action)]], device=device, dtype=th.long)

    def learn(self):
        if self.iter_num % target_update_iter == 0:
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num += 1

        batch = self.replay_mem.sample()
        if len(batch) == 0:
            return

        b_s = th.FloatTensor(np.vstack(batch[:, 0])).to(device)
        b_a = th.LongTensor(batch[:, 1].astype(int).tolist()).to(device)
        b_r = th.FloatTensor(np.vstack(batch[:, 2])).to(device)
        b_s_ = th.FloatTensor(np.vstack(batch[:, 3])).to(device)
        b_d = th.FloatTensor(np.vstack(batch[:, 4])).to(device)
        q_target = th.zeros((batch_size, 1)).to(device)
        q_eval = self.eval_q_net(b_s)
        q_eval = th.gather(q_eval, dim=1, index=th.unsqueeze(b_a, 1))
        q_next = self.target_q_net(b_s_).detach()
        for i in range(b_d.shape[0]):
            if int(b_d[i].tolist()[0]) == 0:
                q_target[i] = b_r[i] + gamma * th.unsqueeze(th.max(q_next[i], 0)[0], 0)
            else:
                q_target[i] = b_r[i]
        td_error = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        self.loss_history.append(td_error.item())

dqn = DQN()

def preprocess_state(obs):
    if isinstance(obs, dict):
        return obs['image'].flatten()
    else:
        return obs.flatten()

for episode in range(1000):
    obs = env.reset()
    s = preprocess_state(obs[0])  # Initial state
    step = 0
    r = 0.0
    total_reward = 0.0
    episode_loss = 0
    episode_value = 0
    done = False
    while not done:
        step += 1
        a = dqn.select_action(th.FloatTensor(s).to(device))
        obs, r, done, terminated, truncatedsa = env.step(a.item())  # 액션을 넘겨줄 때 item() 메서드 사용

        s_ = preprocess_state(obs)
        transition = [s.tolist(), a.item(), [r], s_.tolist(), [done]]
        done = terminated or truncated
        dqn.replay_mem.store_transition(transition)
        total_reward += r
        s = s_
        print("step:", step, "reward:", r)
        print("episode : ", episode, "reward : ", r)
        if dqn.replay_mem.size() > batch_size:
            dqn.learn()

        if done:
            break

    episode_loss = np.mean(dqn.loss_history[-step:]) if step > 0 else 0
    avg_q_value = th.mean(dqn.eval_q_net(th.FloatTensor([s]).to(device))).item()
    wandb.log({
         "episode": episode,
         "reward": total_reward,
         "average_loss": episode_loss,
        "avg_q_value": avg_q_value
     })

th.save(dqn.eval_q_net.state_dict(), "dqn_eval_q_net_min.pth")
th.save(dqn.target_q_net.state_dict(), "dqn_target_q_net_min.pth")

env.close()
wandb.finish()
