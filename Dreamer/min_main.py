from envs.GymMoreRedBalls import GymMoreRedBalls

# env = GymMoreRedBalls(room_size=10, render_mode='human')
env = GymMoreRedBalls(room_size=10) # render_mode 를 human 으로 한 위와 같이하면 실제로 창에 어떻게 행동하는지가 디스플레이됨.
env.reset(seed=123)

for i in range(1000):
	action = env.action_space.sample()
	observation, reward, done, truncated, info = env.step(action)
	print(f"step={i}, action={action}, observation={observation}, reward={reward}, done={done}, info={info}")
	print("env.instr.s_done:", env.instrs.s_done)

	if done:
		env.reset(seed=123)
		break