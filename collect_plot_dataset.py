
import numpy as np
import gym
import gym_miniworld
import pickle
import time
import matplotlib.pyplot as plt
xs = np.arange(-6.5, 6.5, .5)
# xs = np.arange(-6.5, -2.5, .5) # part 1
# xs = np.arange(-2.5, 0.5, .5) # part 2
# xs = np.arange(0.5, 3.0, .5) # part 3
# xs = np.arange(3.0, 6.5, .5) # part 4
ys = np.arange(-6.5, 6.5, .5)

xs= [xs[15]]
buffer = []

env = gym.make("MiniWorld-FourRooms-v0",)
			# agent_pos=[0, 0, 0], obs_view='topview')
for x in xs:
	for y in ys:
		print(x, y)

		# env.set_agent_pos([y, 0, x])
		obs = env.reset()
		# env.render('pyglet', )
		buffer.append(obs)
		plt.imshow(obs)
		plt.show()
		import pdb;pdb.set_trace()
		plt.savefig(f"plots/4r_{y}_{x}.png")

		env.close()

# filename = f'4rooms_textures_plotting_1st_person.pkl'
# file = open(filename, 'wb')
# pickle.dump(buffer, file)

