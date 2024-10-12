from gym.envs.registration import register

register(
    id='chakra-v0',
    entry_point='chakra.envs:ChakraEnv',
    max_episode_steps=200
)

register(
    id='VishamC-v0',
    entry_point='chakra.envs:VishamCEnv',
    max_episode_steps=200
)
