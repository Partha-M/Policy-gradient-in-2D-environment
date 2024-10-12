#!/usr/bin/env python

import click
import numpy as np
import gym
import chakra


def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)


def include_bias(ob):
    return np.append(ob, 1)


# ***********************************UNCOMMENT FOR VISHAMC**********************************
# @click.command()
# @click.argument("env_id", type=str, default="VishamC-v0")
# def main(env_id):
#     # Register the environment
#     rng = np.random.RandomState(42)
#
#     if env_id == 'VishamC-v0':
#         # from rlpa2 import chakra
#         env = gym.make('VishamC-v0')
#         get_action = chakra_get_action
#         obs_dim = env.observation_space.shape[0]
#         action_dim = env.action_space.shape[0]
#     else:
#         raise ValueError(
#             "Unsupported environment: must be 'VishamC' ")


# ************************* UNCOMMENT FOR CHAKRA********************************************
@click.command()
@click.argument("env_id", type=str, default="chakra-v0")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra-v0':
        # from rlpa2 import chakra
        env = gym.make('chakra-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)

    # Initialize parameters
    # ****************** FINAL VALUE OF THETA FOR CHAKRA***********************************

    theta = np.array([[-17.50489398, -0.0934462, -0.07491875],    # UNCOMMENT FOR CHAKRA
                      [-0.1526158, -17.66515993, 0.03056671]])

    # ****************** FINAL VALUE OF THETA FOR VISHAMC***********************************
    # theta = np.array([[-5.19339945e+00, 3.43873506e-02, -1.36560944e-01],     # UNCOMMENT FOR VISHAMC
    #                   [-4.17138345e-02, -2.43173858e+01, -1.24982045e-02]])

    b = 50
    while b != 0:
        ob = env.reset()
        done1 = False
        # Only render the first trajectory
        # Collect a new trajectory
        rewards = []
        while not done1:
            action = get_action(theta, ob, rng=rng)
            if done1:
                break
            next_ob, rew, done1, _ = env.step(action)
            ob = next_ob
            env.render()
            rewards.append(rew)
        b = b - 1
        print("Episode reward: %.2f" % np.sum(rewards))

    env.close()


if __name__ == "__main__":
    main()
