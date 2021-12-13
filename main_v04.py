import sys
import os
from pathlib import Path

sys.path.append("fly/")
import myfly_env_v03 as env
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pygame
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from torch.autograd import Variable


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

OBSERVE = 100
EXPLORE = 2000000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
BATCH_SIZE = 32
GAMMA = 0.99
UPDATE_TIME = 100
REPLAY_MEMORY = 50000
FRAME_PER_ACTION = 1
ax = []
ay = []

WHITE = (255, 255, 255)
RED = (255, 0, 0)
pygame.init()
curr_path = Path.cwd()
curr_path_2 = Path.cwd()
startIMG = pygame.image.load(curr_path / 'bg.png')
startimg = pygame.transform.smoothscale(startIMG, (SCREEN_WIDTH, SCREEN_HEIGHT))
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
font_addr = pygame.font.get_default_font()
font = pygame.font.Font(font_addr, 36)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.current_device()
a = torch.cuda.is_available()
print(a)


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    # 将图像由一种颜色空间转换成另一种颜色空间，图像缩放，输入图像，输出图像大小80,80，单通道灰度图
    ret, mask = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    # 利用THRESH_BINARY进行二值化
    return np.reshape(observation, (1, 80, 80))


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),  # 改变输入数据的值，存储地址相同
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 多维tensor展平成一维
        x = self.fc1(x)
        return self.out(x)



class BrainDQNMain(object):
    def __init__(self, actions):
        self.replayMemory = deque()
        self.timeStep = 0
        self.batch_size = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.Q_net = DeepQNetwork()
        #self.Q_netT = DeepQNetwork()
        self.load()
        #self.loss_func = nn.MSELoss()
        #LR = 1e-3
        #self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

    def load(self):
        if os.path.exists(curr_path_2/'params30.pth'):
            #print("load model param successful")
            self.Q_net.load_state_dict(torch.load(curr_path_2/'params30.pth'))

    def set_perception(self, nextObservation):
        newState = np.append(self.currentState[1:, :, :], nextObservation, axis=0)
        self.currentState = newState

    def get_action(self):
        currentState = torch.Tensor([self.currentState])
        QValue = self.Q_net(currentState)[0]
        #print("QValue",QValue)
        #print("self.Q_net",self.Q_net(currentState))
        action = np.zeros(self.actions)
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                # 生成0-1的随机数
                action_index = random.randrange(self.actions)
                # print("choose random action " + str(action_index))
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue.detach().numpy())

                action[action_index] = 1
        else:
            action[0] = 1

        #if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
          #  self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        # print("qqq",action)
        return action

    def set_init_state(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=0)
        # stack在对应的维度上进行堆叠
        # print(self.currentState.shape)


class Button(object):
    def __init__(self, text, color, x=None, y=None, **kwargs):
        self.surface = font.render(text, True, color)
        self.WIDTH = self.surface.get_width()
        self.HEIGHT = self.surface.get_height()

        if 'centered_x' in kwargs and kwargs['centered_x']:
            self.x = SCREEN_WIDTH // 2 - self.WIDTH // 2
        else:
            self.x = x

        if 'centered_y' in kwargs and kwargs['centered_y']:
            self.y = SCREEN_HEIGHT // 2 - self.HEIGHT // 2
        else:
            self.y = y

    def display(self):
        screen.blit(self.surface, (self.x, self.y))

    def check_click(self, position):
        x_match = self.x < position[0] < self.x + self.WIDTH
        y_match = self.y < position[1] < self.y + self.HEIGHT

        if x_match and y_match:
            return True
        else:
            return False


def starting_screen():

    screen.blit(startimg, (0, 0))

    game_title = font.render('Choose your game style', True, WHITE)

    screen.blit(game_title, (SCREEN_WIDTH // 2 - game_title.get_width() // 2, 150))

    play_button = Button('Manual', RED, None, 350, centered_x=True)
    exit_button = Button('Automatic', WHITE, None, 400, centered_x=True)
    Play_button = Button('Play', RED, None , 450, centered_x=True)

    play_button.display()
    exit_button.display()
    Play_button.display()

    pygame.display.update()

    while True:

        if play_button.check_click(pygame.mouse.get_pos()):
            play_button = Button('Manual', RED, None, 350, centered_x=True)
        else:
            play_button = Button('Manual', WHITE, None, 350, centered_x=True)

        if exit_button.check_click(pygame.mouse.get_pos()):
            exit_button = Button('Automatic', RED, None, 400, centered_x=True)
        else:
            exit_button = Button('Automatic', WHITE, None, 400, centered_x=True)

        if Play_button.check_click(pygame.mouse.get_pos()):
            Play_button = Button('Play', RED, None , 450 , centered_x=True)
        else:
            Play_button = Button('Play', WHITE, None , 450 , centered_x=True)

        play_button.display()
        exit_button.display()
        Play_button.display()

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        if pygame.mouse.get_pressed()[0]:
            if play_button.check_click(pygame.mouse.get_pos()):
                g_type = "manual"
                return g_type
                break
            if exit_button.check_click(pygame.mouse.get_pos()):
                g_type = "auto"
                return g_type
                break
            if Play_button.check_click(pygame.mouse.get_pos()):
                g_type = "play"
                return g_type
                break


def main():

    game_type = starting_screen()
    actions = 3  # 动作个数
    brain = BrainDQNMain(actions)
    fly = env.GameState()
    action0 = np.array([1, 0, 0])
    observation0, reward0, terminal = fly.frame_step(action0, game_type)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.set_init_state(observation0)

    while 1 != 0 :

                action = brain.get_action()
                nextObservation, reward, terminal = fly.frame_step(action, game_type)
                nextObservation = preprocess(nextObservation)
                brain.set_perception(nextObservation)

if __name__ == '__main__':
    main()
