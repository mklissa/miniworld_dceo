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
        self.action_space = spaces.Discrete(self.actions.move_back+1) # 0-indexed
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


class FourRoomsActions(FourRooms):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def step(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        rand = self.rand if self.domain_rand else None
        fwd_step = self.params.sample(rand, 'forward_step')
        fwd_drift = self.params.sample(rand, 'forward_drift')
        turn_step = self.params.sample(rand, 'turn_step')

        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_back:
            self.turn_agent(turn_step * 2)
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_left:
            self.turn_agent(turn_step)
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_right:
            self.turn_agent(-turn_step)
            self.move_agent(fwd_step, fwd_drift)

        # Pick up an object
        elif action == self.actions.pickup:
            # Position at which we will test for an intersection
            test_pos = self.agent.pos + self.agent.dir_vec * 1.5 * self.agent.radius
            ent = self.intersect(self.agent, test_pos, 1.2 * self.agent.radius)
            if not self.agent.carrying:
                if isinstance(ent, Entity):
                    if not ent.is_static:
                        self.agent.carrying = ent

        # Drop an object being carried
        elif action == self.actions.drop:
            if self.agent.carrying:
                self.agent.carrying.pos[1] = 0
                self.agent.carrying = None

        # If we are carrying an object, update its position as we move
        if self.agent.carrying:
            ent_pos = self._get_carry_pos(self.agent.pos, self.agent.carrying)
            self.agent.carrying.pos = ent_pos
            self.agent.carrying.dir = self.agent.dir

        # Generate the current camera image
        if self.obs_view == 'agent':
            obs = self.render_obs()
        else:
            obs = self.render_top_view()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        return obs, reward, done, {}


