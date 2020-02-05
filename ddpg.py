import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import gym

import utils
import run


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exploration_noise(action, random_process, eps):
    return action + eps*random_process.sample().astype(np.float32)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def ddpg(agent, env, args):
    agent.to(device)

    # initialize target networks
    target_agent = type(agent)()
    target_agent.to(device)
    hard_update(target_agent.actor, agent.actor)
    hard_update(target_agent.critic, agent.critic)

    random_process = utils.OrnsteinUhlenbeckProcess(size=env.action_space.shape, sigma=args.sigma, theta=args.theta)
    eps = args.eps_start

    buffer = utils.ReplayBuffer(args.buffer_size)
    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=args.critic_lr)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.actor_lr)

    save_dir = utils.make_process_dirs('ddpg_run')

    # use warmp up steps to add random transitions to the buffer
    state = env.reset()
    done = False
    for _ in range(args.warmup_steps):
        if done: state = env.reset(); done = False
        rand_action = env.action_space.sample()
        next_state, reward, done, info = env.step(rand_action)
        buffer.push(state, rand_action, reward, next_state, done)
        state = next_state

    for episode in range(args.num_episodes):
        state = env.reset()
        random_process.reset_states()
        done = False 
        for step in range(args.max_episode_steps):
            if done: break

            # collect new experience
            action = agent(state)
            noisy_action = exploration_noise(action, random_process, eps)
            next_state, reward, done, info = env.step(noisy_action)
            if args.render: env.render()
            buffer.push(state, noisy_action, reward, next_state, done)
            state = next_state
            batch = buffer.sample(args.batch_size)
            # batch will be None if not enough experience has been collected yet
            if not batch:
                continue
            
            # prepare transitions for models
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            
            cat_tuple = lambda t : torch.cat(t).to(device)
            list_to_tensor = lambda t : torch.tensor(t).unsqueeze(0).to(device)
            state_batch = cat_tuple(state_batch)
            next_state_batch = cat_tuple(next_state_batch)
            action_batch = cat_tuple(action_batch)
            reward_batch = list_to_tensor(reward_batch).T
            done_batch = list_to_tensor(done_batch).T

            # critic update
            target_action_s2 = target_agent.actor(next_state_batch)
            target_action_value_s2 = target_agent.critic(next_state_batch, target_action_s2)
            td_target = reward_batch + args.gamma*(1.-done_batch)*target_action_value_s2
            agent_critic_pred = agent.critic(state_batch, action_batch)
            critic_loss = F.mse_loss(td_target, agent_critic_pred)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # actor update
            agent_actions = agent.actor(state_batch)
            actor_loss = -agent.critic(state_batch, agent_actions).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # move target model towards training model
            soft_update(target_agent.actor, agent.actor, args.tau)
            soft_update(target_agent.critic, agent.critic, args.tau)
            eps = max(args.eps_final, eps - (args.eps_start - args.eps_final)/args.eps_anneal)
        
        if episode % args.eval_interval == 0:
            agent.eval()
            returns = run.run(agent, env, args.eval_episodes, args.max_episode_steps, verbosity=0)
            mean_return = returns.mean()
            print(f"Episodes of training: {episode+1}, mean reward in test mode: {mean_return}")
            agent.train()
    
    agent.save(save_dir)
    return agent

def parse_args():

    parser = argparse.ArgumentParser(description='Train agent with DDPG')
    parser.add_argument('-env', type=str, default='Pendulum-v0', help='training environment')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='number of episodes for training')
    parser.add_argument('--max_episode_steps', type=int, default=250,
                        help='maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='training batch size')
    parser.add_argument('--tau', type=float, default=.001,
                        help='for model parameter % update')
    parser.add_argument('--actor_lr', type=float, default=1e-4,
                        help='actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-3,
                        help='critic learning rate')
    parser.add_argument('--gamma', type=float, default=.99,
                        help='gamma, the discount factor')
    parser.add_argument('--eps_start', type=float, default=1.)
    parser.add_argument('--eps_final', type=float, default=1e-3)
    parser.add_argument('--eps_anneal', type=float, default=1e6)
    parser.add_argument('--theta', type=float, default=.15,
        help='theta for Ornstein Uhlenbeck process computation')
    parser.add_argument('--sigma', type=float, default=.2,
        help='sigma for Ornstein Uhlenbeck process computation')
    parser.add_argument('--buffer_size', type=int, default=100000,
        help='replay buffer size')
    parser.add_argument('--eval_interval', type=int, default=15,
        help='how often to test the agent without exploration (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10,
        help='how many episodes to run for when testing')
    parser.add_argument('--warmup_steps', type=int, default=1000,
        help='warmup length, in steps')
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--env', type=str)
    return parser.parse_args()



if __name__ == "__main__":
    from agent import PendulumAgent, MountaincarAgent

    args = parse_args()

    if args.env == 'Pendulum-v0':
        agent = PendulumAgent()
        env = gym.make(args.env)
    elif args.env == 'MountainCarContinuous-v0':
        agent = MountaincarAgent()
        env = gym.make(args.env)
    else:
        print(f"CL Arg --env {args.env} not recognized")
        exit(1)

    agent = ddpg(agent, env, args)


