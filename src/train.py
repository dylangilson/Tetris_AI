"""
* Dylan Gilson
* dylan.gilson@outlook.com
* March 1, 2023
"""

from custom_tensorboard import CustomTensorBoard
from datetime import datetime
from deep_q_learning_agent import DQNAgent
from keras.engine.saving import save_model
from statistics import mean
from tetris import Tetris
from tqdm import tqdm


# run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(), n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size, discount=discount,
                     replay_start_size=replay_start_size)

    log_dir = f'../checkpoints/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}' \
              f'-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            best_action = None

            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.hard_drop([best_action[0], 0], best_action[1], render=render)
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # checkpoints
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)

    save_model(agent.model, f'{log_dir}/model.hdf', overwrite=True, include_optimizer=True)


if __name__ == '__main__':
    dqn()
    exit(0)
