import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class TMazeEnv(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=140,
            frame_rate=6,
            **kwargs
        )

    def _gen_world(self):
        room1 = self.add_rect_room(
            min_x=-1, max_x=8,
            min_z=-2, max_z=2
        )
        room2 = self.add_rect_room(
            min_x=8, max_x=12,
            min_z=-8, max_z=8
        )
        room1.add_portal(0, min_z=-2, max_z=2)
        room2.add_portal(2, min_z=-2, max_z=2)

        # Add a box at a random end of the hallway
        z_pos = self.rand.elem([room2.min_z + 0.5, room2.max_z - 0.5])
        self.box = Box([room2.mid_x, 0, z_pos], 0, size=0.8, color='red')
        room2.entities.append(self.box)

        # TODO: need method to place_agent and avoid wall/object intersections

        # Choose a random room and position to spawn in
        if self.rand.bool():
            self.agent.pos = np.array([
                self.rand.float(room1.min_x + 0.5, room1.max_x - 0.5),
                0,
                self.rand.float(room1.min_z + 0.5, room1.max_z - 0.5)
            ])
            self.agent.dir = self.rand.float(-math.pi/3, math.pi/3)
        else:
            self.agent.pos = np.array([
                self.rand.float(room2.min_x + 0.5, room2.max_x - 0.5),
                0,
                self.rand.float(room2.min_z + 1.5, room2.max_z - 1.5)
            ])
            self.agent.dir = self.rand.float(-math.pi/2, math.pi/2)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # TODO: proper intersection test method for entities
        # Entity.pos_inside(p)?
        dist = np.linalg.norm(self.agent.pos - self.box.pos)
        if dist < self.box.size:
            reward += self._reward()
            done = True

        return obs, reward, done, info