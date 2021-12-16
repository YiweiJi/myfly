# coding:utf-8

"""
@author:lisheng
Created on 2021/12/15 14:15
"""

import random
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import Game


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 3
        self.gamma = 0.99
        self.final_epsilon = 0.05
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 750000
        self.minibatch_size = 32
        self.explore = 3000000

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


def preprocessing(image):
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    image_tensor = image_data.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    criterion = nn.MSELoss()

    game_state = Game()

    D = deque()

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 0
    image_data, reward, terminal = game_state.step(action)
    image_data = preprocessing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    epsilon = model.initial_epsilon
    iteration = 0

    while iteration < model.number_of_iterations:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        random_action = random.random() <= epsilon
        if random_action:
            print("Random action!")

        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        action[action_index] = 1

        if epsilon > model.final_epsilon:
            epsilon -= (model.initial_epsilon - model.final_epsilon) / model.explore

        image_data_1, reward, terminal = game_state.step(action)
        image_data_1 = preprocessing(image_data_1)

        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        D.append((state, action, reward, state_1, terminal))

        if len(D) > model.replay_memory_size:
            D.popleft()

        minibatch = random.sample(D, min(len(D), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
        # print("state_1_batch size: ", state_1_batch.shape)

        output_1_batch = model(state_1_batch)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        # print("q_value: ", q_value.shape)

        optimizer.zero_grad()

        y_batch = y_batch.detach()

        loss = criterion(q_value, y_batch)

        loss.backward()
        optimizer.step()

        state = state_1
        iteration += 1

        if iteration % 10000 == 0:
            torch.save(model, "trained_model/current_model_" + str(iteration) + ".pth")

        print("total iteration: {} Elapsed time: {:.2f} epsilon: {:.5f}"
              " action: {} Reward: {:.1f}".format(iteration, ((time.time() - start) / 60), epsilon,
                                                  action_index.cpu().detach().numpy(), reward.numpy()[0][0]))


def test(model):
    game_state = Game()

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.step(action)
    image_data = preprocessing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        action_index = torch.argmax(output)
        action[action_index] = 1

        image_data_1, reward, terminal = game_state.step(action)
        image_data_1 = preprocessing(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1


def main():
    model = NeuralNetwork()
    model.apply(init_weights)
    start = time.time()
    train(model, start)


if __name__ == "__main__":
    main()
