import gym


class Environment(object):
    def __init__(self, env_name='MsPacman-v0'):
        self.env = gym.make(env_name)

    def play(self, episodes, time_steps=None):
        pass


class Agent(object):
    def __init__(self):
        pass
