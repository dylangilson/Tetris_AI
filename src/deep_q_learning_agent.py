"""
* Dylan Gilson
* dylan.gilson@outlook.com
* March 1, 2023
"""

from collections import deque
from keras import Model
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
from typing import List, Optional, ValuesView


# Deep Q Learning Agent + Maximin
# this version only provides only value per input indicating the score expected in that state
# this is because the algorithm will try to find the best final state for the combinations of possible states,
# instead of the traditional way of finding the best action for a particular state.
class DQNAgent:
    """
    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Optimizer used
        replay_start_size: Minimum size needed to train
    """

    def __init__(self, state_size, mem_size=10000, discount=0.95, epsilon=1, epsilon_min=0, epsilon_stop_episode=500,
                 n_neurons=(32, 32), activations=('relu', 'relu', 'linear'), loss='mse', optimizer='adam',
                 replay_start_size=None):
        assert len(activations) == len(n_neurons) + 1

        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer

        if not replay_start_size:
            replay_start_size = mem_size / 2

        self.replay_start_size = replay_start_size
        self.model = self.build_model()

    # builds a keras deep neural network model
    def build_model(self) -> Model:
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    # adds a play to the replay memory buffer
    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))

    # assigns a random score for a certain action
    def random_value(self):
        return random.random()

    # predicts the score for a certain state
    def predict_value(self, state: np.ndarray) -> float:
        return self.model.predict(state)[0]

    # return the expected score of a certain state
    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)

    def best_state(self, states: ValuesView[List[int]]) -> List[int]:
        """Returns the best state for a given collection of states"""
        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            max_value: Optional[float] = None
            best_state: Optional[List[int]] = None

            for state in states:
                # ask the neural network about the best value
                value = self.predict_value(np.reshape(state, [1, self.state_size]))

                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, batch_size=32, epochs=3):
        n = len(self.memory)

        if n >= self.replay_start_size and n >= batch_size:
            batch = random.sample(self.memory, batch_size)

            # get the expected score for the next states as a batch
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.model.predict(next_states)]

            x = []
            y = []

            # build xy structure to fit the model as a batch
            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    new_q = reward + self.discount * next_qs[i]  # partial Q formula
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            # fit the model to the given values
            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

            # update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
