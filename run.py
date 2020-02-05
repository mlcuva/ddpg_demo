import torch
import utils

def run(agent, env, episodes, max_steps, render=False, verbosity=1):
    episode_return_history = []
    for episode in range(episodes):
        episode_return = 0
        state = env.reset()
        done, info = False, {}
        for _ in range(max_steps):
            if done: break
            action = agent.forward(state)
            state, reward, done, info = env.step(action)
            if render: env.render()
            episode_return += reward
        if verbosity:
            print(f"Episode {episode}:: {episode_return}")
        episode_return_history.append(episode_return)
    return torch.tensor(episode_return_history)