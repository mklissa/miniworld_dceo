# import sys
# import argparse
# import pyglet

# from pyglet.window import key
# from pyglet import clock
import numpy as np
import gym
import gym_miniworld
import pickle


env = gym.make("MiniWorld-FourRooms-v0")
buffer = []
num_actions = 4
t = 0
while t < 2e5:

	episode = []
	obs = env.reset()
	done = False
	episode.append(obs)

	while not done:
		t += 1
		obs, _, done, _ = env.step(np.random.randint(num_actions))
		episode.append(obs)
	buffer.append(episode)
	if t  % 5e4 == 0:
		print(t)
		filename = f'4rooms_textures_buffer_1st_person_{t}.pkl'
		file = open(filename, 'wb')
		pickle.dump(buffer, file)
		buffer = []
# import pdb;pdb.set_trace()


