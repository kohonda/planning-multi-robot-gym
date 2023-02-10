import math
import pygame

# TODO: move these to a config
WIDTH = 1500
HEIGHT = 1000
size = [WIDTH, HEIGHT]
black = (20, 20, 40)
lightblue = (0, 120, 255)
darkblue = (0, 40, 160)
red = (255, 100, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
grey = (70, 70, 70)
k = 160
u0 = WIDTH / 2
v0 = HEIGHT / 2

class Robot:
    def __init__(self, x: float, y, theta: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.vL = 0.0
        self.vR = 0.0
        self.path = None
        self.location_history = []

    def draw(self, screen: pygame.Surface, robot_radius: float, wheel_blob: float):
        robot_width = 2 * robot_radius
        u = u0 + k * self.x
        v = v0 - k * self.y
        pygame.draw.circle(screen, white, (int(u), int(v)), int(k * robot_radius), 3)

        wlx = self.x - (robot_width / 2.0) * math.sin(self.theta)
        wly = self.y + (robot_width / 2.0) * math.cos(self.theta)
        ulx = u0 + k * wlx
        vlx = v0 - k * wly
        pygame.draw.circle(screen, blue, (int(ulx), int(vlx)), int(k * wheel_blob))

        wrx = self.x + (robot_width / 2.0) * math.sin(self.theta)
        wry = self.y - (robot_width / 2.0) * math.cos(self.theta)
        urx = u0 + k * wrx
        vrx = v0 - k * wry
        pygame.draw.circle(screen, blue, (int(urx), int(vrx)), int(k * wheel_blob))

        if self.path is not None:
            if self.path[0] == 0:
                straight_path = self.path[1]
                line_start = (u0 + k * self.x, v0 - k * self.y)
                line_end = (
                    u0 + k * (self.x + straight_path * math.cos(self.theta)),
                    v0 - k * (self.y + straight_path * math.sin(self.theta)),
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
