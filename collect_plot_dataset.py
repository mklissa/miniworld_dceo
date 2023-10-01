
import numpy as np
import gym
import gym_miniworld
import pickle
import os
import time
import matplotlib.pyplot as plt


# MyWayHome values
ys = np.arange(-10.3, 10.7, .25)
xs = np.arange(-4.3, 14.7, .25)

# # FourRooms values
xs = np.arange(-6.5, 6.5, .5)
ys = np.arange(-6.5, 6.5, .5)

buffer = []
valid_pos = []

plot_directory = 'plots/'
os.makedirs(plot_directory, exist_ok=True)
data_directory = 'datasets/'
os.makedirs(data_directory, exist_ok=True)

env = gym.make("MiniWorld-FourRoomsCoverage-v0")
default_obs = env.reset()

pos_dict = {}
iter = 0
i = 0
for i, x in enumerate(xs):
	for j, y in enumerate(ys):
		pos_dict[f'ij'] = (x, y)
		print(x, y)
		env.set_agent_pos([y, 0, x])
		obs = env.reset()

		# cur_valid = valid_pos[i] 
		# if (i > 0 and i  < len(valid_pos) - 1) and (valid_pos[i-1] and valid_pos[i+1]):
		# 	cur_valid = True

		if not env.valid_pos:
			obs = default_obs
		valid_pos.append(env.valid_pos)

		env.render(view='top')
		time.sleep(0.05)
		plt.imshow(obs)
		plt.savefig(f"plots/4r_{iter}_{y:.2f}_{x:.2f}_{env.valid_pos}.png")

		i += 1
		iter += 1
		buffer.append(obs)

env.close()

height = len(xs)
width = len(ys)

data = {
	'images': buffer,
	'valid_pos': valid_pos,
	'frame_size': (height, width),
	'pos_dict': pos_dict,
}

filename = f'datasets/4r_again_plotting_1st_person.pkl'
with open(filename, 'wb') as f:
	pickle.dump(data, f)

