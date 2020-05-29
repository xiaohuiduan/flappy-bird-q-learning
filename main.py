import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
import random


class Agent():

    def __init__(self, action_space):

        # 获得游戏支持的动作集合
        self.action_set = action_space

        # 创建q-table
        self.q_table = np.zeros((6, 6, 6, 2))

        # 学习率
        self.alpha = 0.7
        self.gamma = 0.8
        self.greedy = 0.8

    def get_state(self, state):
        """
        提取游戏state中我们需要的数据
        :param state: 游戏state
        :return: 返回提取好的数据
        """
        return_state = np.zeros((3,), dtype=int)
        dist_to_pipe_horz = state["next_pipe_dist_to_player"]
        dist_to_pipe_bottom = state["player_y"] - state["next_pipe_top_y"]
        velocity = state['player_vel']
        if velocity < -15:
            velocity_category = 0
        elif velocity < -10:
            velocity_category = 1
        elif velocity < -5:
            velocity_category = 2
        elif velocity < 0:
            velocity_category = 3
        elif velocity < 5:
            velocity_category = 4
        else:
            velocity_category = 5

        if dist_to_pipe_bottom < 8:  # very close or less than 0
            height_category = 0
        elif dist_to_pipe_bottom < 20:  # close
            height_category = 1
        elif dist_to_pipe_bottom < 50:  # not close
            height_category = 2
        elif dist_to_pipe_bottom < 125:  # mid
            height_category = 3
        elif dist_to_pipe_bottom < 250:  # far
            height_category = 4
        else:
            height_category = 5

        # make a distance category
        if dist_to_pipe_horz < 8:  # very close
            dist_category = 0
        elif dist_to_pipe_horz < 20:  # close
            dist_category = 1
        elif dist_to_pipe_horz < 50:  # not close
            dist_category = 2
        elif dist_to_pipe_horz < 125:  # mid
            dist_category = 3
        elif dist_to_pipe_horz < 250:  # far
            dist_category = 4
        else:
            dist_category = 5

        return_state[0] = height_category
        return_state[1] = dist_category
        return_state[2] = velocity_category
        return return_state

    def update_q_table(self, old_state, current_action, next_state, r):
        """

        :param old_state: 执行动作前的状态
        :param current_action: 执行的动作
        :param next_state: 执行动作后的状态
        :param r: 奖励
        :return:
        """
        next_max_value = np.max(self.q_table[next_state[0], next_state[1], next_state[2]])

        self.q_table[old_state[0], old_state[1], old_state[2], current_action] = (1 - self.alpha) * self.q_table[
            old_state[0], old_state[1], old_state[2], current_action] + self.alpha * (r + next_max_value)

    def get_best_action(self, state, greedy=False):
        """
        获得最佳的动作
        :param state: 状态
        :return: 最佳动作
        """

        jump = self.q_table[state[0], state[1], state[2], 0]
        no_jump = self.q_table[state[0], state[1], state[2], 1]

        if greedy:
            if np.random.rand(1) < self.greedy:
                return np.random.choice([0, 1])
            else:
                if jump > no_jump:
                    return 0
                else:
                    return 1
        else:
            if jump > no_jump:
                return 0
            else:
                return 1

    def update_greedy(self):
        self.greedy *= 0.95

    def act(self, p, action):
        """
        执行动作
        :param p: 通过p来向游戏发出动作命令
        :param action: 动作
        :return: 奖励
        """
        # action_set表示游戏动作集(119，None)，其中119代表跳跃
        r = p.act(self.action_set[action])
        if r == 0:
            r = 1
        if r == 1:
            r = 10
        else:
            r = -1000
        return r


if __name__ == "__main__":
    episodes = 2000_000000
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=False)
    p.init()
    agent = Agent(p.getActionSet())
    max_score = 0

    for episode in range(episodes):
        p.reset_game()
        state = agent.get_state(game.getGameState())
        agent.update_greedy()
        while True:
            action = agent.get_best_action(state)
            reward = agent.act(p, action)
            next_state = agent.get_state(game.getGameState())
            agent.update_q_table(state, action, next_state, reward)
            current_score = p.score()
            state = next_state
            if p.game_over():
                max_score = max(current_score, max_score)
                print('Episodes: %s, Current score: %s, Max score: %s' % (episode, current_score, max_score))
                if current_score > 300:
                    np.save("{}_{}.npy".format(current_score, episode), agent.q_table)
                break
