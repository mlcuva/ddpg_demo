import argparse

import gym
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


if __name__ == "__main__":
    from agent import PendulumAgent, MountaincarAgent
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', type=int, default=1)
    parser.add_argument('--env', type=str)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--agent', type=str)
    parser.add_argument('--max_steps', type=int, default=300)
    args = parser.parse_args()

    if args.env == 'Pendulum-v0':
        agent = PendulumAgent()
        env = gym.make(args.env)
    elif args.env == 'MountainCarContinuous-v0':
        agent = MountaincarAgent()
        env = gym.make(args.env)
    else:
        print(f"CL Arg --env {args.env} not recognized")
        exit(1)
    
    agent.load(args.agent)
    run(agent, env, args.episodes, args.max_steps, args.render, verbosity=1)
