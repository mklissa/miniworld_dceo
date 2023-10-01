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

    def __init__(self, agent_pos=[-6.5, 0, -6.5], coverage_plot=False, **kwargs):
        self.agent_pos = agent_pos
        self.coverage_plot = coverage_plot
        super().__init__(
            max_episode_steps=1000,
            **kwargs
        )
        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_back+1)
        agent_x = np.random.uniform(-6, -2)
        agent_z = np.random.uniform(-6, -2)
        self.agent_pos = [agent_x, 0., agent_z]

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1 , max_z=7,
            wall_tex='cardboard'
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7,
            wall_tex='marble'
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1 , max_x=7,
            min_z=-7, max_z=-1,
            wall_tex='metal_grill'
        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1,
            wall_tex='stucco'
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        self.box = self.place_entity(Box(color='red'), pos=np.array([1.5, 0.0, 1.5]))
        self.place_agent(dir=0, pos=self.agent_pos)

        if self.coverage_plot:
            conds = []
            for room in self.rooms:
                conds.append(room.point_inside(self.agent_pos))
            self.valid_pos = any(conds)

    def set_agent_pos(self, agent_pos):
        self.agent_pos = agent_pos

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info


class FourRoomsCoverage(FourRooms):
    def __init__(self):
        super().__init__(coverage_plot=True)