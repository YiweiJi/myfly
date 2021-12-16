import sys
import os
from pathlib import Path

sys.path.append("fly/")
import environment as env
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
paths = ["images"]
# curr_path = Path.cwd()
curr_path = Path.cwd().joinpath(*paths)
curr_path_2 = Path.cwd().joinpath(*paths)
startIMG = pygame.image.load(curr_path / 'bg.png')
startimg = pygame.transform.smoothscale(startIMG, (SCREEN_WIDTH, SCREEN_HEIGHT))
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
font_addr = pygame.font.get_default_font()
font = pygame.font.Font(font_addr, 36)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.current_device()
# a = torch.cuda.is_available()
# print(a)

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
        x = x.to(device)
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
        self.Q_net = DeepQNetwork().to(device)
        self.Q_netT = DeepQNetwork().to(device)
        self.load()
        self.loss_func = nn.MSELoss().to(device)
        LR = 1e-3
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

    def save(self):
        #print("save model param")
        # state_dict = torch.load("params3.pth")
        torch.save(self.Q_net.state_dict(), curr_path_2/'params31.pth')  # 保存训练好的参数
        # torch.save(state_dict, 'params3.pth',_use_new_zipfile_serialization=False)

    def load(self):
        if os.path.exists(curr_path_2/'params31.pth'):
            #print("load model param successful")
            self.Q_net.load_state_dict(torch.load(curr_path_2/'params31.pth'))
            self.Q_netT.load_state_dict(torch.load(curr_path_2/'params31.pth'))

    def train(self):
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        y_batch = np.zeros([BATCH_SIZE, 1])
        nextState_batch = np.array(nextState_batch)
        nextState_batch = torch.Tensor(nextState_batch)
        action_batch = np.array(action_batch)

        index = action_batch.argmax(axis=1)
        # axis=1是在行中比较选出最大的列索引，axis=0是在列中比较选出最大的行索引。
        index = np.reshape(index, [BATCH_SIZE, 1])
        action_batch_tensor = torch.LongTensor(index)

        QValue_batch = self.Q_netT(nextState_batch)  # 使用target网络，预测nextState_batch的动作
        QValue_batch = QValue_batch.detach().cpu().numpy()
        # 计算每个state的reward
        total_reward = 0
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                y_batch[i][0] = reward_batch[i] + GAMMA * np.max(QValue_batch[i])
            total_reward += reward_batch[i]


        each_reward = total_reward / BATCH_SIZE
        each_reward = round(each_reward, 2)
        ay.append(each_reward)
        # print("qqqq",each_reward)
        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [BATCH_SIZE, 1])
        # print('bbbbbbbbbbbb', each_reward)
        state_batch_tensor = Variable(torch.Tensor(state_batch))
        # variable提供了自动求导的功能
        y_batch_tensor = Variable(torch.Tensor(y_batch))
        y_predict = self.Q_net(state_batch_tensor).gather(1, action_batch_tensor)  # 索引对应Q值
        loss = self.loss_func(y_predict, y_batch_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()

    def set_perception(self, nextObservation, action, reward, terminal):
        newState = np.append(self.currentState[1:, :, :], nextObservation, axis=0)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            self.train()
            self.batch_size += 1
            ax.append(self.batch_size)
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        # print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
        self.currentState = newState
        self.timeStep += 1

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
                action_index = np.argmax(QValue.detach().cpu().numpy())

                action[action_index] = 1
        else:
            action[0] = 1

        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
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
    # Step 1: init BrainDQN


    game_type = starting_screen()
    actions = 3  # 动作个数
    brain = BrainDQNMain(actions)
    fly = env.Game()
    action0 = np.array([1, 0, 0])
    observation0, reward0, terminal = fly.step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.set_init_state(observation0)

    while 1 != 0 :

                action = brain.get_action()
                nextObservation, reward, terminal = fly.step(action)
                nextObservation = preprocess(nextObservation)
                brain.set_perception(nextObservation, action, reward, terminal)

if __name__ == '__main__':
    main()
