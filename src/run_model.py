"""
* Dylan Gilson
* dylan.gilson@outlook.com
* March 7, 2023
"""

import cv2
from deep_q_learning_agent import DQNAgent
from keras.engine.saving import load_model
import os
from tetris import Tetris


def run_model(dir_name, episodes=100, render=False):
    env = Tetris()
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    replay_start_size = 2000
    n_neurons = [32, 32]
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(), n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size, discount=discount,
                     replay_start_size=replay_start_size)

    model_path = '../checkpoints/' + dir_name + '/model.hdf'
    agent.model = load_model(model_path)
    agent.epsilon = 0
    scores = []

    for episode in range(episodes):
        env.reset()
        game_over = False

        while not game_over:
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            # find the action, that corresponds to the best state
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            _, game_over = env.hard_drop([best_action[0], 0], best_action[1], render)

        scores.append(env.score)

        print(f'episode {episode} => {env.score}')
    return scores


def run_model_helper(episodes=128, render=False):
    dirs = [name for name in os.listdir('../checkpoints') if os.path.isdir(os.path.join('../checkpoints', name))]
    dirs.sort(reverse=True)
    dirs = [dirs[0]]

    max_scores = []
    for directory in dirs:
        print(f"Evaluating dir '{directory}'")
        scores = run_model(directory, episodes, render)
        max_scores.append((directory, max(scores)))

    max_scores.sort(key=lambda t: t[1], reverse=True)

    for k, v in max_scores:
        print(f"{v}\t{k}")


if __name__ == '__main__':
    run_model_helper(16, True)
    cv2.destroyAllWindows()
    exit(0)
