from typing import NamedTuple
import math
import pygame


class Robot:
    def __init__(self, x: float, y, theta: float, graphics: NamedTuple):
        self.x = x
        self.y = y
        self.theta = theta
        self.vL = 0.0
        self.vR = 0.0
        self.path = None
        self.location_history = []
        self.graphics = graphics

    def draw(self, screen: pygame.Surface, robot_radius: float, wheel_blob: float):
        robot_width = 2 * robot_radius
        u = self.graphics.center[0] + self.graphics.scale * self.x
        v = self.graphics.center[1] - self.graphics.scale * self.y
        pygame.draw.circle(
            screen,
            self.graphics.white,
            (int(u), int(v)),
            int(self.graphics.scale * robot_radius),
            3,
        )

        wlx = self.x - (robot_width / 2.0) * math.sin(self.theta)
        wly = self.y + (robot_width / 2.0) * math.cos(self.theta)
        ulx = self.graphics.center[0] + self.graphics.scale * wlx
        vlx = self.graphics.center[1] - self.graphics.scale * wly
        pygame.draw.circle(
            screen,
            self.graphics.blue,
            (int(ulx), int(vlx)),
            int(self.graphics.scale * wheel_blob),
        )

        wrx = self.x + (robot_width / 2.0) * math.sin(self.theta)
        wry = self.y - (robot_width / 2.0) * math.cos(self.theta)
        urx = self.graphics.center[0] + self.graphics.scale * wrx
        vrx = self.graphics.center[1] - self.graphics.scale * wry
        pygame.draw.circle(
            screen,
            self.graphics.blue,
            (int(urx), int(vrx)),
            int(self.graphics.scale * wheel_blob),
        )

        if self.path is not None:
            if self.path[0] == 0:
                straight_path = self.path[1]
                line_start = (
                    self.graphics.center[0] + self.graphics.scale * self.x,
                    self.graphics.center[1] - self.graphics.scale * self.y,
                )
                line_end = (
                    self.graphics.center[0]
                    + self.graphics.scale
                    * (self.x + straight_path * math.cos(self.theta)),
                    self.graphics.center[1]
                    - self.graphics.scale
                    * (self.y + straight_path * math.sin(self.theta)),
                )
                pygame.draw.line(screen, (0, 200, 0), line_start, line_end, 1)
            if self.path[0] == 2:
                if self.path[3] > self.path[2]:
                    start_angle = self.path[2]
                    stop_angle = self.path[3]
                else:
                    start_angle = self.path[3]
                    stop_angle = self.path[2]
                if start_angle < 0:
                    start_angle += 2 * math.pi
                    stop_angle += 2 * math.pi
                if (
                    self.path[1][1][0] > 0
                    and self.path[1][0][0] > 0
                    and self.path[1][1][1] > 1
                ):
                    pygame.draw.arc(
                        screen, (0, 200, 0), self.path[1], start_angle, stop_angle, 1
                    )
