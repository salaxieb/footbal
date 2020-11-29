import arcade
from arcade.buffered_draw_commands import _Batch
import math
import numpy as np
from collections import defaultdict
import tensorflow as tf
from scipy.special import softmax
from conf import *

class Players():
    def __init__(self):
        super().__init__()

        self.weights = self.brain_builder(init_weight=0.2)
        self.test_weights = {vec: np.copy(self.weights[vec]) for vec in self.weights}
        self.effective_gradient = self.brain_builder(init_weight=0)
        self.new_shift = self.brain_builder(init_weight=0)

        self.gamma = 0.999
        self.pointer = 0
        self.limit = 3200
        self.history_view = 1
        self.history = np.array([[np.zeros(shape=N_SENSORS + 2 * 7), np.nan, np.nan, np.nan]]*self.limit)
        self.filled_initial_examples = False
        self.rewards = [0]
        self.current_reward = 0
        self.setup()

    def draw(self):
        for p in self.players:
            p.draw()

    def setup(self):
        self.p1 = Player(SCREEN_WIDTH/2 - 100, SCREEN_HEIGHT/2 - 100, team = 'blue')
        self.p2 = Player(SCREEN_WIDTH/2 - 100, SCREEN_HEIGHT/2 + 100, team = 'blue')

        self.p3 = Player(SCREEN_WIDTH/2 + 100, SCREEN_HEIGHT/2 - 100, team = 'red')
        self.p4 = Player(SCREEN_WIDTH/2 + 100, SCREEN_HEIGHT/2 + 100, team = 'red')

        self.team_blue = [self.p1, self.p2]
        self.team_red = [self.p3, self.p4]
        self.players = self.team_blue + self.team_red

        self.score_blue = 0
        self.score_red = 0

        last_mean_reward = np.mean(self.rewards[-min(len(self.rewards), self.history_view):])
        print('curr:', self.current_reward, 'last mean:', last_mean_reward)

        if last_mean_reward < 0 and self.current_reward < 0:
            coefficient = math.log(last_mean_reward/self.current_reward)
        elif self.current_reward == 0:
            if last_mean_reward <= 0:
                coefficient = 1
            else:
                coefficient = -1
        elif last_mean_reward <= 0:
            if self.current_reward > 0:
                coefficient = 20
            else:
                coefficient = 1
        elif self.current_reward <= 0 and last_mean_reward >= 0:
            coefficient = -1
        else:
            coefficient = math.log(self.current_reward/last_mean_reward)
        print('coeff', coefficient)

        for vec_name in self.effective_gradient:
            #self.effective_gradient[vec_name] += coefficient*self.new_shift[vec_name]
            self.weights[vec_name] += 0.1/max(1, math.log(len(self.rewards)))*coefficient*self.new_shift[vec_name]

        self.new_shift = self.brain_builder(init_weight=0.1)
        self.test_weights = {vec: np.copy(self.weights[vec])+self.new_shift[vec] for vec in self.weights}

        self.rewards.append(round(self.current_reward))
        self.current_reward = 0
        print(len(self.rewards), self.rewards[-min(len(self.rewards), 10):])

    def update(self, delta_time, ball, map):
        self.score_blue += (self.p1.holding_ball-self.p1.on_wall)*delta_time
        self.score_blue += (self.p2.holding_ball-self.p2.on_wall)*delta_time

        self.score_red += (self.p3.holding_ball-self.p3.on_wall)*delta_time
        self.score_red += (self.p4.holding_ball-self.p4.on_wall)*delta_time

        s1 = self.p1.update(delta_time, map.walls, ball, map.gate_b, map.gate_r, self.p2, self.p3, self.p4)
        s2 = self.p2.update(delta_time, map.walls, ball, map.gate_b, map.gate_r, self.p1, self.p3, self.p4)

        s3 = self.p3.update(delta_time, map.walls, ball, map.gate_r, map.gate_b, self.p4, self.p1, self.p2)
        s4 = self.p4.update(delta_time, map.walls, ball, map.gate_r, map.gate_b, self.p3, self.p1, self.p2)

        states = np.concatenate(([s1], [s2], [s3], [s4]), axis=0)

        actions = self.predict(states)
        for p, pred in zip(self.players, actions):
            p.act(pred, delta_time)

        self.current_reward += sum([p.holding_ball - p.on_wall for p in self.players]) * delta_time * 60

        # for s, a, r, a_v in zip(states, actions, rewards, actions_values):
        #     self.history[self.pointer] = np.array([s, a, r, a_v], dtype='object')
        #     self.pointer += 1
        #     if self.pointer >= self.limit:
        #         self.pointer = 0
        #         self.filled_initial_examples = True

    def brain_builder(self, init_weight=0.1):
        weights = {
        'W1': np.random.randn(N_SENSORS + 2 * 7 + 1, 20) * init_weight,

        'W2': np.random.randn(21, 10) * init_weight,

        'W3': np.random.randn(11, 5) * init_weight,

        'W4': np.random.randn(6, 2) * init_weight
        }
        return weights

    @staticmethod
    def sigmoid(Z):
        return 1-2/(1+np.exp(-Z))

    @staticmethod
    def elu(Z):
        return np.where(Z>0, Z, np.exp(Z)-1)

    def predict(self, inp):
        Z1 = np.dot(np.concatenate((inp, np.ones((4, 1))), axis=1), self.test_weights['W1'])
        Z1 = self.elu(Z1)

        Z2 = np.dot(np.concatenate((Z1, np.ones((4, 1))), axis=1), self.test_weights['W2'])
        Z2 = self.elu(Z2)

        Z3 = np.dot(np.concatenate((Z2, np.ones((4, 1))), axis=1), self.test_weights['W3'])
        Z3 = self.elu(Z3)

        Z4 = np.dot(np.concatenate((Z3, np.ones((4, 1))), axis=1), self.test_weights['W4'])
        output = self.sigmoid(Z4)
        return output

    def propogate(self, mutation_weights):
        for vec in self.weights:
            self.weights[vec] += mutation_weights[vec]

    # def train_on_batch(self):
    #     if self.filled_initial_examples:
    #         batch = np.random.randint(0, self.limit-1, size=32)
    #         next_b = batch + 1
    #         inp = np.array([s for s, a, r, a_v in self.history[batch]])
    #         rewards = np.array([r for s, a, r, a_v in self.history[batch]])
    #         actions = np.array([a for s, a, r, a_v in self.history[batch]])
    #         action_values = np.array([a_v for s, a, r, a_v in self.history[next_b]])
    #         action_values = self.gamma * action_values
    #         action_values[:, actions] = action_values[:, actions] + rewards
    #         self.brain.train_on_batch(inp, action_values)

    def check_for_collision_with_ball(self, ball):
        for p in self.players:
            p.check_for_collision_with_ball(ball)

class Player(arcade.ShapeElementList):
    def __init__(self, center_x, center_y, team='blue', size=20, eyesight=300):
        super().__init__()
        self.team = team
        if self.team == 'blue':
            self.color = (40, 40, 200)
            self.g_color = (40, 40, 100)
        elif self.team == 'red':
            self.color = (200, 40, 40)
            self.g_color = (100, 40, 40)
        else:
            self.color = (200, 200, 200)
            self.g_color = (100, 100, 100)

        self.center_x = center_x
        self.center_y = center_y

        self.holding_ball = False
        self.on_wall = False

        self.speed = 150
        self.radius = size/2
        self.eyesight = eyesight
        self.sensors = np.array([[0, 0, eyesight*math.cos(a), eyesight*math.sin(a)] for a in np.linspace(0, 2*math.pi, N_SENSORS)])

        self.actions = {0: (1, 0),
                        1: (1/math.sqrt(2), 1/math.sqrt(2)),
                        2: (0, 1),
                        3: (-1/math.sqrt(2), 1/math.sqrt(2)),
                        4: (-1, 0),
                        5: (-1/math.sqrt(2), -1/math.sqrt(2)),
                        6: (0, -1),
                        7: (1/math.sqrt(2), -1/math.sqrt(2))}

    def act(self, pred, delta_time):
        #action = np.random.choice(range(len(pred)), p=softmax(pred))
        #direction = self.actions[action]
        self.change_x = self.speed * pred[0] * delta_time
        self.change_y = self.speed * pred[1] * delta_time
        self.move(self.change_x, self.change_y)
        #return action

    def update(self, delta_time, walls, ball, my_gate, enemy_gate, team_player, enemy_1, enemy_2):
        self.batches = defaultdict(_Batch)
        distances = np.min(self.distances(walls), axis=1)
        cut_sensors = np.multiply(np.resize(distances, (N_SENSORS, 1)), self.sensors)

        for [sx, sy, ex, ey] in cut_sensors:
            self.append(arcade.create_line(sx, sy, ex, ey, color=self.g_color))

        circ = arcade.create_ellipse_filled(0, 0, width=20, height=20, color=self.color)

        self.append(circ)
        self.append(circ)

        self.holding_ball = self.check_for_collision_with_ball(ball)
        self.on_wall = np.min(distances) < 0.05

        return self.generate_input(distances, ball, my_gate, enemy_gate, team_player, enemy_1, enemy_2)

    def generate_input(self, distances, ball, my_gate, enemy_gate, team_player, enemy_1, enemy_2):
        norm = 200
        return np.concatenate((
            distances,
            [(ball.center_x-self.center_x)/norm, (ball.center_y-self.center_y)/norm],
            [(team_player.center_x-self.center_x)/norm, (team_player.center_y-self.center_y)/norm],
            [(enemy_1.center_x-self.center_x)/norm, (enemy_1.center_y-self.center_y)/norm],
            [(enemy_2.center_x-self.center_x)/norm, (enemy_2.center_y-self.center_y)/norm],
            [(my_gate.center_x-self.center_x)/norm, (my_gate.center_y-self.center_y)/norm],
            [(enemy_gate.center_x-self.center_x)/norm, (enemy_gate.center_y-self.center_y)/norm],
            [self.holding_ball, team_player.holding_ball]
        ))

    def check_for_collision_with_ball(self, ball):
        if self.radius + ball.radius > np.linalg.norm([self.center_x - ball.center_x, self.center_y - ball.center_y]):
            return self.radius + ball.radius / max(5, np.linalg.norm([self.center_x - ball.center_x, self.center_y - ball.center_y]))
        else:
            return 0

    @property
    def vision_rays(self):
        return self.sensors + [[self.center_x, self.center_y, self.center_x, self.center_y]]*N_SENSORS

    def distances(self, walls):
        sensor = self.vision_rays
        walls = np.repeat([walls], N_SENSORS, axis=0)
        #http://www.cs.swan.ac.uk/~cssimon/line_intersection.html
        x1 = sensor[:,0]
        y1 = sensor[:,1]
        x2 = sensor[:,2]
        y2 = sensor[:,3]
        x3 = walls[:,:,0]
        y3 = walls[:,:,1]
        x4 = walls[:,:,2]
        y4 = walls[:,:,3]

        a_top = (y3-y4)*np.transpose(np.subtract(x1,np.transpose(x3)))+(x4-x3)*np.transpose(np.subtract(y1,np.transpose(y3)))
        a_bottom = np.transpose(np.multiply(np.transpose(x4-x3),(y1-y2)))-np.transpose(np.multiply(np.transpose(y4-y3),(x1-x2)))
        b_top = np.transpose(np.multiply(np.subtract(x1,np.transpose(x3)),(y1-y2)))+np.transpose(np.multiply(np.subtract(y1,np.transpose(y3)),(x2-x1)))
        b_bottom = a_bottom
        sensor_cross_walls = a_top/a_bottom
        walls_cross_sensor = b_top/b_bottom

        sensor_cross_walls[sensor_cross_walls<0] = 1
        sensor_cross_walls[sensor_cross_walls>1] = 1
        sensor_cross_walls[walls_cross_sensor<0] = 1
        sensor_cross_walls[walls_cross_sensor>1] = 1
        return sensor_cross_walls #result [0;1] if sensor crosses wall
