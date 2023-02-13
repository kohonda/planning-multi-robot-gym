from typing import Optional, Tuple, Any, List, Dict, Union

import copy
import gymnasium as gym
import math
import numpy as np
import pygame
import os
import yaml
from collections import namedtuple
from gymnasium import spaces
from numpy import ndarray

from planning_multi_robot_gym.robot import Robot


class PlanningMultiRobotEnv(gym.Env):
    """A multi-robot planning environment for gym.

    This environment simulates the movements of multiple robots in an environment with barriers and a target. The robots
    are controlled by the actions given by an agent. The observations received by the agent includes information about
    the position, velocity and orientation of each robot, the future position of the target and the future position of
    the obstacles. The goal of the agent is to navigate the robots to the target while avoiding collisions with the
    obstacles.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        n_robots: int = 5,
        n_barriers: int = 20,
        render_mode: Optional[str] = None,
        barrier_radius: float = 0.1,
        robot_radius: float = 0.1,
        wheel_blob: float = 0.04,
        max_velocity: float = 0.5,
        max_acceleration: float = 0.4,
        barrier_velocity_range: float = 0.2,
        dt: float = 0.1,
        steps_ahead_to_plan: int = 10,
        reach_target_reward: float = 1000.0,
        collision_penalty: float = -500.0,
        reset_when_target_reached: bool = False,
        config: Dict = None,
    ):
        """
        Args:
            n_robots (int): The number of robots in the environment.
            n_barriers (int): The number of barriers in the environment.
            render_mode (str): The render mode of the environment, either "rgb_array" or "human".
            barrier_radius (float): The radius of each barrier.
            robot_radius (float): The radius of each robot.
            wheel_blob (float): The size of the wheel blob.
            max_velocity (float): The maximum velocity of each robot.
            max_acceleration (float): The maximum acceleration of each robot.
            barrier_velocity_range (float): The velocity range of each barrier.
            dt (float): The time step of the simulation.
            steps_ahead_to_plan (int): The number of steps ahead the robots should plan for.
            reach_target_reward (float): The reward given when a robot reaches the target.
            collision_penalty (float): The penalty given when a robot collides with a barrier or another robot.
            reset_when_target_reached (bool): A flag indicating whether the environment should reset when a robot
                reaches the target.
            config (dict): The graphics configuration of the environment.
        """
        self.robots: List[Robot] = []
        self.target_index = None
        self.barriers = None
        self.n_barriers = n_barriers
        self.n_robots = n_robots
        self.barrier_radius = barrier_radius
        self.robot_radius = robot_radius
        self.robot_width = robot_radius * 2
        self.dt = dt
        self.steps_ahead_to_plan = steps_ahead_to_plan
        self.tau = dt * self.steps_ahead_to_plan
        self.reach_target_reward = reach_target_reward
        self.collision_penalty = collision_penalty
        self.reset_when_target_reached = reset_when_target_reached
        self.wheel_blob = wheel_blob
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.barrier_velocity_range = barrier_velocity_range

        # load graphics config
        if config is None:
            # Default graphics config
            path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = yaml.safe_load(open(path, "r"))
        else:
            if not isinstance(config, Dict):
                raise ValueError("config must be a dictionary")

        graphics_config = config["graphics"]
        graphics_config["center"] = [
            graphics_config["window_width"] / 2,
            graphics_config["window_height"] / 2,
        ]

        GraphicConfig = namedtuple("GraphicConfig", graphics_config.keys())
        self.graphics = GraphicConfig(**graphics_config)

        play_field_config = config["play_field"]
        PlayField = namedtuple("PlayField", play_field_config.keys())
        self.play_field = PlayField(**play_field_config)
        # self.play_field_corners: Tuple[float, float, float, float] = (
        #     play_field_config["min_x"],
        #     play_field_config["min_y"],
        #     play_field_config["max_x"],
        #     play_field_config["max_y"],
        # )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Dict(
            {
                "vR": spaces.Box(
                    low=-self.max_velocity,
                    high=self.max_velocity,
                    shape=(n_robots,),
                    dtype=float,
                ),
                "vL": spaces.Box(
                    low=-self.max_velocity,
                    high=self.max_velocity,
                    shape=(n_robots,),
                    dtype=float,
                ),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "vR": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_robots,), dtype=float
                ),
                "vL": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_robots,), dtype=float
                ),
                "theta": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_robots,), dtype=float
                ),
                "robot_positions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_robots, 2), dtype=float
                ),
                "future_target_position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=float
                ),
                "future_obstacle_positions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_barriers - 1, 2), dtype=float
                ),
            }
        )

        self._info = {
            "n_robots": self.n_robots,
            "max_acceleration": self.max_acceleration,
            "max_velocity": self.max_velocity,
            "predict_position": self._predict_position,
            "barrier_radius": self.barrier_radius,
            "robot_radius": self.robot_radius,
            "dt": self.dt,
            "tau": self.tau,
        }

        self.window = None
        self.clock = None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        barriers = copy.deepcopy(self.barriers)
        for _ in range(self.steps_ahead_to_plan):
            self._move_barriers(barriers)
        return {
            "vR": np.array([robot.vR for robot in self.robots]),
            "vL": np.array([robot.vL for robot in self.robots]),
            "theta": np.array([robot.theta for robot in self.robots]),
            "robot_positions": np.array([(robot.x, robot.y) for robot in self.robots]),
            "future_target_position": np.array(
                [barriers[self.target_index][0], barriers[self.target_index][1]]
            ),
            "future_obstacle_positions": np.array(
                [
                    (barrier[0], barrier[1])
                    for i, barrier in enumerate(barriers)
                    if i != self.target_index
                ]
            ),
        }

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[Dict[str, Any], None] = None
    ):
        super().reset(seed=seed)

        barriers_list = []
        for i in range(self.n_barriers):
            barrier = [
                self.np_random.uniform(
                    self.play_field.min_x, self.play_field.max_x
                ),
                self.np_random.uniform(
                    self.play_field.min_y, self.play_field.max_y
                ),
                self.np_random.normal(0.0, self.barrier_velocity_range),
                self.np_random.normal(0.0, self.barrier_velocity_range),
            ]
            barriers_list.append(barrier)
        self.barriers = np.array(barriers_list)

        self.target_index = self.np_random.integers(0, self.n_barriers)

        self.robots = [
            Robot(
                x=self.play_field.min_x - 0.5,
                y=-2.0 + 0.8 * i,
                theta=0.0,
                graphics=self.graphics,
            )
            for i in range(self.n_robots)
        ]

        return self._get_obs(), self._info

    def _move_barriers(self, barriers) -> None:        
        # update obstacles positions
        barriers[:, 0] += barriers[:, 2] * self.dt
        barriers[:, 1] += barriers[:, 3] * self.dt
        
        # if obstacles hit the walls, change direction
        barriers[:, 2] = np.where(
            (barriers[:, 0] < self.play_field.min_x) | (barriers[:, 0] > self.play_field.max_x),
            -barriers[:, 2],
            barriers[:, 2]
        )
        barriers[:, 3] = np.where(
            (barriers[:, 1] < self.play_field.min_y) | (barriers[:, 1] > self.play_field.max_y),
            -barriers[:, 3],
            barriers[:, 3]
        )

    def _predict_position(self, vL, vR, x, y, theta, deltat):
        if round(vL, 3) == round(vR, 3):
            xnew = x + vL * deltat * math.cos(theta)
            ynew = y + vL * deltat * math.sin(theta)
            thetanew = theta
            path = (0, vL * deltat)
        elif round(vL, 3) == -round(vR, 3):
            xnew = x
            ynew = y
            thetanew = theta + ((vR - vL) * deltat / self.robot_width)
            path = (1, 0)
        else:
            R = self.robot_width / 2.0 * (vR + vL) / (vR - vL)
            deltatheta = (vR - vL) * deltat / self.robot_width
            xnew = x + R * (math.sin(deltatheta + theta) - math.sin(theta))
            ynew = y - R * (math.cos(deltatheta + theta) - math.cos(theta))
            thetanew = theta + deltatheta

            (cx, cy) = (x - R * math.sin(theta), y + R * math.cos(theta))
            Rabs = abs(R)
            ((tlx, tly), (Rx, Ry)) = (
                (
                    int(self.graphics.center[0] + self.graphics.scale * (cx - Rabs)),
                    int(self.graphics.center[1] - self.graphics.scale * (cy + Rabs)),
                ),
                (
                    int(self.graphics.scale * (2 * Rabs)),
                    int(self.graphics.scale * (2 * Rabs)),
                ),
            )
            if R > 0:
                start_angle = theta - math.pi / 2.0
            else:
                start_angle = theta + math.pi / 2.0
            stop_angle = start_angle + deltatheta
            path = (2, ((tlx, tly), (Rx, Ry)), start_angle, stop_angle)

        return xnew, ynew, thetanew, path

    def step(self, action: Dict[str, ndarray]):
        assert self.action_space.contains(action)

        self._move_barriers(self.barriers)

        reward = 0
        for i, robot in enumerate(self.robots):
            robot.location_history.append((robot.x, robot.y))
            vL = min(
                max(robot.vL - self.max_acceleration * self.dt, action["vL"][i]),
                robot.vL + self.max_acceleration * self.dt,
            )
            vR = min(
                max(robot.vR - self.max_acceleration * self.dt, action["vR"][i]),
                robot.vR + self.max_acceleration * self.dt,
            )

            (robot.x, robot.y, robot.theta, _) = self._predict_position(
                vL, vR, robot.x, robot.y, robot.theta, self.dt
            )
            (_, _, _, robot.path) = self._predict_position(
                vL, vR, robot.x, robot.y, robot.theta, self.tau
            )
            robot.vL = vL
            robot.vR = vR

        robot_has_reached_target = False
        
        # check for collisions and reach target
        for robot in self.robots:
            # calculate distances to all barriers
            dists = np.sqrt((robot.x - self.barriers[:, 0]) ** 2 + (robot.y - self.barriers[:, 1]) ** 2)
            # get indices of barriers that collide with the robot
            collision_indices = np.where(dists < (self.barrier_radius + self.robot_radius))[0]
            if self.target_index in collision_indices:
                robot_has_reached_target = True
                reward += self.reach_target_reward
            else:
                reward += self.collision_penalty

        if robot_has_reached_target:
            self.target_index = self.np_random.integers(0, self.n_barriers)

            for robot in self.robots:
                robot.location_history = []

        observation = self._get_obs()
        terminated = self.reset_when_target_reached and robot_has_reached_target
        truncated = False  # No time limit

        return (
            observation,
            reward,
            terminated,
            truncated,
            self._info,
        )

    def _draw_barriers(self, screen):
        for i, barrier in enumerate(self.barriers):
            if i == self.target_index:
                bcol = self.graphics.red
            else:
                bcol = self.graphics.lightblue
            pygame.draw.circle(
                screen,
                bcol,
                (
                    int(self.graphics.center[0] + self.graphics.scale * barrier[0]),
                    int(self.graphics.center[1] - self.graphics.scale * barrier[1]),
                ),
                int(self.graphics.scale * self.barrier_radius),
                0,
            )

    def render(self) -> Optional[ndarray]:
        if self.render_mode:
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.graphics.window_width, self.graphics.window_height)
                )
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface(
                (self.graphics.window_width, self.graphics.window_height)
            )
            canvas.fill(self.graphics.black)

            for robot in self.robots:
                for loc in robot.location_history:
                    pygame.draw.circle(
                        surface=canvas,
                        color=self.graphics.grey,
                        center=(
                            int(self.graphics.center[0] + self.graphics.scale * loc[0]),
                            int(self.graphics.center[1] - self.graphics.scale * loc[1]),
                        ),
                        radius=3,
                        width=0,
                    )

            self._draw_barriers(canvas)

            for robot in self.robots:
                robot.draw(canvas, self.robot_radius, self.wheel_blob)

            if self.render_mode == "human":
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()

                self.clock.tick(self.metadata["render_fps"])
            else:
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
