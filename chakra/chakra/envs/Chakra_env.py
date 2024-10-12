from gym import Env
from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
import numpy as np

# ******************** CHAKRA ENVIRONMENT ********************************

class ChakraEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
# ************************INITIALIZATION********************************

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))

        self._seed()
        self.viewer = None
        self.state = None
        self.reward = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
# *************** STEP ACTION DECLARATION*****************

    def _step(self, action):
        self.state += 0.025*action
        if np.array([np.greater(self.state, 1), np.less(self.state, -1)]).any():  # If agent crosses boundary end
            self.reward = -np.linalg.norm(self.state)                             # episode and give negative reward
            self.reset()
        elif (np.less(self.state, 0.0125) & np.greater(self.state, -0.0125)).all():
            self.reward = 0                                                      # if near to origin give zero reward
            self.done = True                                                     # and stop episode
        else:
            self.reward = -np.linalg.norm(self.state)                            # reward equal to negative of distance
            self.done = False                                                    # from origin

        return np.array(self.state), self.reward, self.done, {}
# ****************************** RESET FUNCTION ***********************

    def _reset(self):
        while True:
            self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
            # Sample states that are far away
            if np.linalg.norm(self.state) > 0.9:
                self.done = False
                break
        return np.array(self.state)

# ***********************RENDERING****************************************

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            origin = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            trans = rendering.Transform(translation=(0, 0))
            agent.add_attr(trans)
            self.trans = trans
            agent.set_color(1, 0, 0)
            origin.set_color(0, 0, 0)
            origin.add_attr(rendering.Transform(
                translation=(screen_width // 2, screen_height // 2)))
            self.viewer.add_geom(agent)
            self.viewer.add_geom(origin)

        # self.trans.set_translation(0, 0)
        self.trans.set_translation(
            (self.state[0] + 1) / 2 * screen_width,
            (self.state[1] + 1) / 2 * screen_height,
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

# ********************************CLOSING******************************************
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
# register(
#     id='chakra-v0',
#     entry_point='rlpa2.chakra:chakra',        # REGISTER IS DECLARED IN __init__.py FILE
#     max_episode_steps = 40,
# )