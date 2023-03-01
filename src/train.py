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
from typing import Optional, Iterable


class AgentConf:
    def __init__(self):
        self.n_neurons = [32, 32]
        self.batch_size = 512
        self.activations = ['relu', 'relu', 'linear']
        self.episodes = 2000
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_stop_episode = 2000
        self.mem_size = 25000
        self.discount = 0.99
        self.replay_start_size = 2000
        self.epochs = 1
        self.render_every = None
        self.train_every = 1
        self.log_every = 10
        self.max_steps: Optional[int] = 10000


# run dqn with Tetris
def dqn(conf: AgentConf):
    env = Tetris()

    agent = DQNAgent(env.get_state_size(), n_neurons=conf.n_neurons, activations=conf.activations,
                     epsilon=conf.epsilon, epsilon_min=conf.epsilon_min, epsilon_stop_episode=conf.epsilon_stop_episode,
                     mem_size=conf.mem_size, discount=conf.discount, replay_start_size=conf.replay_start_size)

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f'checkpoints/tetris-{timestamp_str}-ms{conf.mem_size}-e{conf.epochs}' \
                     f'-ese{conf.epsilon_stop_episode}-d{conf.discount}'
    log = CustomTensorBoard(log_dir=checkpoint_dir)

    print(f'AGENT_CONF = {checkpoint_dir}')

    scores = []

    episodes_wrapped: Iterable[int] = tqdm(range(conf.episodes))
    for episode in episodes_wrapped:
        current_state = env.reset()
        done = False
        steps = 0

        # update render flag
        render = True if conf.render_every and episode % conf.render_every == 0 else False

        # game
        while not done and (not conf.max_steps or steps < conf.max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            # find the action, that corresponds to the best state
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.hard_drop([best_action[0], 0], best_action[1], render=render)

            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())  # return score

        # train
        if episode % conf.train_every == 0:
            agent.train(batch_size=conf.batch_size, epochs=conf.epochs)

        # checkpoints
        if conf.log_every and episode and episode % conf.log_every == 0:
            avg_score = mean(scores[-conf.log_every:])
            min_score = min(scores[-conf.log_every:])
            max_score = max(scores[-conf.log_every:])
            log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)

    # save_model
    save_model(agent.model, f'{checkpoint_dir}/model.hdf', overwrite=True, include_optimizer=True)


if __name__ == '__main__':
    conf = AgentConf()
    dqn(conf)
    exit(0)
