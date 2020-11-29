import arcade
import math
import numpy as np
from conf import *

class Map(arcade.ShapeElementList):
    def __init__(self, width=1000, height=600):
        super().__init__()

        color = (255, 255, 255)
        gate_width = 100
        gate_height = 300

        self.width = width
        self.width = height

        self.points = [
                  [-width/2, gate_height/2, -width/2, height/2],
                  [-width/2, height/2, width/2, height/2],
                  [width/2, height/2, width/2, gate_height/2],
                  [-width/2, -gate_height/2, -width/2, -height/2],
                  [-width/2, -height/2, width/2, -height/2],
                  [width/2, -height/2, width/2, -gate_height/2]]

        for [s_x, s_y, e_x, e_y] in self.points:
            self.append(arcade.create_line(s_x, s_y, e_x, e_y, color=color, line_width=5))

        points = [(-width/2-gate_width, -gate_height/2),
                  (-width/2-gate_width, +gate_height/2),
                  (-width/2, +gate_height/2),
                  (-width/2, -gate_height/2)]
        self.gate_b = arcade.create_rectangle_filled_with_colors(
                                                point_list=points,
                                                color_list=[(40,40,200)]*len(points))

        self.gate_b.center_x = SCREEN_WIDTH/2-width/2-gate_width/2
        self.gate_b.center_y = SCREEN_HEIGHT/2

        points = [(width/2+gate_width, -gate_height/2),
                  (width/2+gate_width, +gate_height/2),
                  (width/2, +gate_height/2),
                  (width/2, -gate_height/2)]
        self.gate_r = arcade.create_rectangle_filled_with_colors(
                                                point_list=points,
                                                color_list=[(200,40,40)]*len(points))

        self.gate_r.center_x = SCREEN_WIDTH/2+width/2+gate_width/2
        self.gate_r.center_y = SCREEN_HEIGHT/2

        self.center_x = SCREEN_WIDTH / 2
        self.center_y = SCREEN_HEIGHT / 2

        self.append(self.gate_b)
        self.append(self.gate_r)
        self.append(arcade.create_line(s_x, s_y, e_x, e_y, color=color, line_width=5))
        self.append(arcade.create_line(s_x, s_y, e_x, e_y, color=color, line_width=5))

    @property
    def walls(self):
        shift = [[self.center_x, self.center_y, self.center_x, self.center_y],
                [self.center_x, self.center_y, self.center_x, self.center_y],
                [self.center_x, self.center_y, self.center_x, self.center_y],
                [self.center_x, self.center_y, self.center_x, self.center_y],
                [self.center_x, self.center_y, self.center_x, self.center_y],
                [self.center_x, self.center_y, self.center_x, self.center_y]]
        return np.array(self.points) + np.array(shift)
