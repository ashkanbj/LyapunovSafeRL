from gym.envs.registration import register

register(
    id="GridNav-v0",
    entry_point="lyapunovrl.utils.gridnavigation:GridNavigationEnv",
    kwargs={"gridsize": 25},
    max_episode_steps=200,
)

register(
    id="GridNav-v1",
    entry_point="lyapunovrl.utils.gridnavigation:GridNavigationEnv",
    kwargs={"gridsize": 60},
    max_episode_steps=200,
)
