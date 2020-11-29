import random
import arcade
import math
import os
import numpy as np
# from bacteria import Bacterias, Bacteria
from conf import *
from ball import Ball
from player import Players
from map import Map


class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self):
        """ Initializer """
        # Call the parent class initializer
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        # Set the working directory (where we expect to find files) to the same
        # directory this .py file is in. You can leave this out of your own
        # code, but it is needed to easily run the examples using "python -m"
        # as mentioned at the top of this program.
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)
        arcade.set_background_color(arcade.color.AMAZON)
        self.players = Players()

    def setup(self):
        """ Set up the game and initialize the variables. """
        self.ball = Ball(200, 200)
        self.players.setup()
        self.map = Map()
        arcade.set_background_color(arcade.color.AMAZON)
        self.timer = 2
        self.score_red = 0
        self.score_blue = 0

    def on_draw(self):
        """ Render the screen. """
        # This command has to happen before we start drawing
        arcade.start_render()
        self.ball.draw()
        self.players.draw()
        self.map.draw()
        arcade.draw_text(str(round(self.timer)), 10, 20, arcade.color.BLACK, 14)
        arcade.draw_text(str(round(self.players.score_red)), self.map.gate_r.center_x, self.map.gate_r.center_y + 200, arcade.color.RED, 30)
        arcade.draw_text(str(round(self.players.score_blue)), self.map.gate_b.center_x, self.map.gate_b.center_y + 200, arcade.color.BLUE, 30)
        arcade.draw_text(str(self.players.pointer), 40, 20, arcade.color.BLACK, 14)

    def on_mouse_press(self, x, y, button, modifiers):
        """ Called whenever the mouse button is clicked. """
        pass

    def on_update(self, delta_time):
        """ Movement and game logic """
        self.timer -= delta_time
        if self.timer < 0:
            self.setup()
        self.players.update(delta_time, self.ball, self.map)



def main():
    game = MyGame()
    game.setup()
    arcade.run()

if __name__ == "__main__":
    main()
