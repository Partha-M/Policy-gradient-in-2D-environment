#!/usr/bin/env python

import click
import numpy as np
import gym
import chakra       # IMPORT vishamC ENVIRONMENT

# **************** ACTION FUNCTION*****************************

def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)
# *************** INCLUDING BIAS TERM***************************

def include_bias(ob):
    return np.append(ob, 1)


@click.command()
@click.argument("env_id", type=str, default="VishamC-v0")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'VishamC-v0':
        # from rlpa2 import chakra
        env = gym.make('VishamC-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'VishamC' ")

    env.seed(42)

    # Initialize parameters

    max_iteration = 100
    iteration = 0
    batch_size = 500
    gamma = 0.9
    alpha_theta = 0.0005                                       # learning rate for updating policy parameters
    alpha_w = 0.001                                            # learning rate for updating baseline (value) parameters
    theta1 = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))          # initialize theta
    while iteration <= max_iteration:

        batch = 0
        reward = []
        action_set = []
        current_state =[]
        next_state = []
        Grad_total = 0                                          # gradient for theta update
        Grad_v_total = 0                                        # gradient for w update
        theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
        w = rng.normal(scale=0.01, size=(1, obs_dim + 1))
        while batch <= batch_size:
            ob = env.reset()                                    # reset environment
            done3 = False
            while not done3:
                action = get_action(theta, ob, rng=rng)
                action_set.append(action)
                next_ob, rew, done3, _ = env.step(action)       # get reward and next state
                ob_1 = np.append(ob, 1)
                current_state.append(ob_1)
                next_ob1 = np.append(next_ob, 1)
                next_state.append(next_ob1)
                ob = next_ob

                reward.append(rew)                              # collect all rewards
                total_return = 0
                Grad = 0
                Grad_v = 0
                T = np.size(reward)
            for i in range(T - 1):
                total_return = reward[T - 1 - i] + gamma * total_return     # calculate return for each state in batch
                v = np.dot(w, np.square(current_state[T-1-i]))              # baseline calculation

                grad = np.dot(np.transpose([action_set[T - 1 - i]]), [next_state[T - 1 - i]]) - np.dot(theta, np.dot(
                    np.transpose([next_state[T - 1 - i]]), [next_state[T - 1 - i]]))  # calculate gradient for theta
                delta = total_return-v                           # advantage calculation
                Grad += delta * grad                             # accumulate gradients
                grad_v = np.square(current_state[T-1-i])         # gradient for value is (s_1^2,s_2^2,1)
                grad_v = grad_v / (np.linalg.norm(grad_v) + 1e-8)    # normalize
                w += alpha_w * delta * grad_v                    # update w
            Grad = Grad / (np.linalg.norm(Grad) + 1e-8)
            Grad_total += Grad
            Grad_v_total += Grad_v
            batch += 1
        ob = env.reset()                                        # testing for updated theta
        done2 = False
        total_reward = 0
        while not done2:
            action = get_action(theta1, ob, rng=rng)
            next_ob, rew, done2, _ = env.step(action)
            ob = next_ob
            total_reward += rew
            # if iteration >= 95:
            #     env.render()
        theta1 += alpha_theta * Grad_total                      # update theta

        print(iteration)
        print(total_reward)
        print(theta1)
        print(w)
        iteration += 1

    env.close()


if __name__ == "__main__":
    main()
