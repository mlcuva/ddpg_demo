import argparse

import gym

from agent import PendulumAgent, MountaincarAgent
from run import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', type=int, default=1)
    parser.add_argument('--env', type=str)
    parser.add_argument('--episodes', type=int)
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


