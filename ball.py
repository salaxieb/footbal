import arcade
import math
import numpy as np
from conf import *

class Ball(arcade.ShapeElementList):
    def __init__(self, center_x, center_y, radius=80, color=(220,220,220), num_segments=-1):
        super().__init__()

        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

        point_list = [(self.radius*math.sin(a), self.radius*math.cos(a)) for a in np.linspace(0, 2*math.pi, 10)]

        my_line_strip = arcade.create_line_loop(point_list, color=color, line_width=2)
        self.append(my_line_strip)

        self.center_x = SCREEN_WIDTH / 2
        self.center_y = SCREEN_HEIGHT / 2
        self.angle = 10
