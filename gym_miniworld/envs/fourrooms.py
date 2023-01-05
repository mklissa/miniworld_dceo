import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class FourRooms(MiniWorldEnv):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def __init__(self, agent_pos=[-6.5, 0, -6.5], **kwargs):
        self.agent_pos = agent_pos
        super().__init__(
            max_episode_steps=250,
            **kwargs
        )
        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_back+1)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1 , max_z=7,
            # wall_tex='cardboard'
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7,
            # wall_tex='marble'
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1 , max_x=7,
            min_z=-7, max_z=-1,
            # wall_tex='metal_grill'
        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1,
            # wall_tex='stucco'
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        # self.box = self.place_entity(Box(color='red'))
        self.place_agent(dir=0)#, pos=self.agent_pos)

    def set_agent_pos(self, agent_pos):
        self.agent_pos = agent_pos

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # if self.near(self.box):
        #     reward += self._reward()
        #     done = True

        return obs, reward, done, info
