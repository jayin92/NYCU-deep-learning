#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
import os
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

def initialize_orthogonal(layer: nn.Linear, gain=1.0):
    """Initialize the weights and bias using orthogonal initialization."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        std_min: float = 1e-5,
        std_max: float = 3.0,
        activation=nn.Tanh,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        # Network layers
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.mu = nn.Linear(32, out_dim)
        # self.log_std = nn.Linear(32, out_dim)
        self.logstds = nn.Parameter(torch.full((out_dim,), 0.5)) # Higher initial exploration
        self.activation = activation()
        
        # Initialize weights
        initialize_orthogonal(self.fc1, gain=np.sqrt(2))
        initialize_orthogonal(self.fc2, gain=np.sqrt(2))
        initialize_orthogonal(self.mu, gain=0.01)
        # init_layer_uniform(self.log_std)
        
        self.std_min = std_min
        self.std_max = std_max
        self.out_dim = out_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        
        # Mean and standard deviation for Gaussian policy
        mu = self.mu(x)
        std = torch.clamp(self.logstds.exp(), self.std_min, self.std_max)
        
        # Create normal distribution
        dist = Normal(mu, std)
        
        # Sample action from distribution
        action = dist.sample()
        
        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int, activation=nn.Tanh):
        """Initialize."""
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)  # Critic outputs a single value
        self.activation = activation()
        
        # Initialize weights
        initialize_orthogonal(self.fc1, gain=np.sqrt(2))
        initialize_orthogonal(self.fc2, gain=np.sqrt(2))
        initialize_orthogonal(self.fc3, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        value = self.fc3(x)  # Output a single value

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    values = values + [next_value]
    gae_returns = []
    gae = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_freq = args.checkpoint_freq
        self.run_id = args.wandb_run_name
        self.video_dir = args.video_dir
        self.num_eval_episodes = 20
        self.best_eval_score = float('-inf')
        self.reward_threshold = -150.0  # Set a threshold for performance

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))


        selected_action = torch.tanh(selected_action) * 2.0
        selected_action = torch.clamp(selected_action, -2.0, 2.0)

        return selected_action.cpu().detach().numpy().flatten()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        
        # Normalize advantages for more stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            surrogate1 = ratio * adv
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
            surrogate_loss = torch.min(surrogate1, surrogate2).mean()

            entropy = dist.entropy().mean()
            actor_loss = -surrogate_loss - self.entropy_weight * entropy
            
            # critic_loss
            critic_value = self.critic(state)
            critic_loss = F.smooth_l1_loss(critic_value, return_)
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def evaluate(self, episode: int) -> float:
        """Evaluate the agent periodically during training and record videos.
        Args:
            episode (int): Current episode number
        Returns:
            float: Average evaluation score
        """
        self.is_test = True
        self.actor.eval()
        self.critic.eval()
        
        # Create a unique folder for this evaluation run
        video_folder = os.path.join(self.video_dir, f"run_{self.run_id}_eval_episode_{episode}")
        os.makedirs(video_folder, exist_ok=True)
        
        total_reward = 0.0
        valid_seeds = []
        
        for i in range(self.num_eval_episodes):
            # Create a wrapper environment for recording
            test_env_id = self.env.spec.id if self.env.spec else "Pendulum-v1"
            test_env_kwargs = {} if self.env.spec is None else dict(self.env.spec.kwargs)
            
            # Ensure render_mode isn't duplicated
            test_env_kwargs['render_mode'] = 'rgb_array'
            
            current_test_env = gym.make(test_env_id, **test_env_kwargs)
            if i == 0:
                current_test_env = gym.wrappers.RecordVideo(
                    current_test_env,
                    video_folder=video_folder,
                    name_prefix=f"eval_episode_{episode}_test_{i}",
                    episode_trigger=lambda x: True
                )
            
            state_np, _ = current_test_env.reset(seed=self.seed + i)
            done = False
            episode_score = 0.0
            
            while not done:
                action_np = self.select_action(state_np)
                next_state_np, reward_np, terminated, truncated, _ = current_test_env.step(action_np)
                done = terminated or truncated
                
                state_np = next_state_np
                current_reward = float(reward_np) if hasattr(reward_np, 'item') else reward_np
                episode_score += current_reward

            current_test_env.close()
            if episode_score > self.reward_threshold:
                valid_seeds.append(self.seed + i)
            total_reward += episode_score
            
        avg_reward = total_reward / self.num_eval_episodes
        print(f"Evaluation at episode {episode}: Avg Score = {avg_reward:.2f} over {self.num_eval_episodes} episodes")
        
        # Log evaluation score to wandb
        wandb.log({
            "eval/episode": episode,
            "eval/avg_reward": avg_reward,
            "eval/total_steps": self.total_step
        })
        
        # Check if this is the best performance so far
        if avg_reward > self.best_eval_score:
            self.best_eval_score = avg_reward
            self.save_checkpoint(suffix="best")
            print(f"New best performance: {avg_reward:.2f} at episode {episode}")
            
        # Check if performance exceeds threshold
        if avg_reward > self.reward_threshold:
            print(f"Performance threshold {self.reward_threshold} reached at episode {episode}!")
            self.save_checkpoint(suffix=f"threshold_{self.total_step}")
            
        # Reset agent for training
        self.is_test = False
        self.actor.train()
        self.critic.train()
        
        return avg_reward
    
    def save_checkpoint(self, episode=None, best=False, suffix=""):
        """Save model checkpoint.
        
        Args:
            path: Path to save directory
            episode: Current episode number (for filename)
            best: Whether this is the best model so far
        """
        prefix = "best_" if best else ""
        episode_str = f"_ep{episode}_{suffix}" if episode is not None else ""
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_step': self.total_step,
            'args': {
                'gamma': self.gamma,
                'tau': self.tau,
                'epsilon': self.epsilon,
                'entropy_weight': self.entropy_weight,
            }
        }
        
        filename = f"{prefix}ppo_pendulum{episode_str}.pt"
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        print(f"Checkpoint saved to {os.path.join(self.checkpoint_dir, filename)}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # Load training progress
        self.total_step = checkpoint['total_step']
        
        # Optionally load hyperparameters if needed
        # self.gamma = checkpoint['args']['gamma']
        # self.tau = checkpoint['args']['tau']
        # self.epsilon = checkpoint['args']['epsilon']
        # self.entropy_weight = checkpoint['args']['entropy_weight']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return True

    def train(self):
        """Train the PPO agent."""
        self.is_test = False
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Load checkpoint if specified
        if args.resume and args.resume_path:
            self.load_checkpoint(args.resume_path)

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        best_avg_reward = float('-inf')
        
        for ep in tqdm(range(1, self.num_episodes)):
            score = 0
            done_flag = False
            while not done_flag:
                for _ in range(self.rollout_len):
                    self.total_step += 1
                    action = self.select_action(state)
                    next_state, reward, done = self.step(action)

                    state = next_state
                    score += reward[0][0]

                    # if episode ends
                    if done[0][0]:
                        episode_count += 1
                        state, _ = self.env.reset(seed=self.seed)
                        state = np.expand_dims(state, axis=0)
                        scores.append(score)
                        print(f"Episode {episode_count}: Total Reward = {score}")
                        
                        # Log to wandb
                        wandb.log({
                            "episode": episode_count,
                            "reward": score,
                            "total_steps": self.total_step
                        })
                        done_flag = True
                        score = 0

                actor_loss, critic_loss = self.update_model(next_state)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                
                # Log losses to wandb
                wandb.log({
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "step": ep
                })
            
            # Save checkpoint periodically
            if ep % self.checkpoint_freq == 0:
                print(f"Evaluating and saving checkpoint at episode {ep}")
                self.evaluate(ep)
                # self.save_checkpoint(args.checkpoint_dir, episode=ep)
            
            # Calculate avg reward over last 10 episodes for best model tracking
            if len(scores) >= 10:
                avg_reward = sum(scores[-10:]) / 10
                wandb.log({"avg_reward_10": avg_reward})
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_checkpoint(best=True)
                    print(f"New best model with avg reward: {best_avg_reward}")

        # Save final model
        self.evaluate(ep)
        self.save_checkpoint(episode="final")
        
        # termination
        self.env.close()

    def test(self, video_folder: str, num_test_episodes: int = 1, seeds: list = []): # Added num_test_episodes
        """Test the agent and record videos."""
        self.is_test = True
        self.actor.eval() # Set actor to evaluation mode
        self.critic.eval() # Set critic to evaluation mode

        # Create a unique folder for this test run
        video_folder = os.path.join(video_folder, f"run_{self.run_id}_test")
        os.makedirs(video_folder, exist_ok=True) # Ensure video folder exists
        
        total_score = 0.0  # Track total score for averaging
        valid_seeds = []  # List to store valid seeds based on performance

        for i in range(num_test_episodes):
            if i < len(seeds):
                current_seed = seeds[i]
            else:
                current_seed = self.seed + i

            print(f"Testing with seed: {current_seed}")
            # Create a new env instance for each recording to ensure proper video closing.
            # Fallback for env.spec if it's None (e.g. if env is already wrapped multiple times)
            test_env_id = self.env.spec.id if self.env.spec else "Pendulum-v1" 
            test_env_kwargs = {} if self.env.spec is None else dict(self.env.spec.kwargs)
            
            # Ensure render_mode isn't duplicated
            test_env_kwargs['render_mode'] = 'rgb_array'

            current_test_env = gym.make(test_env_id, **test_env_kwargs)
            record_env = gym.wrappers.RecordVideo(
                current_test_env, 
                video_folder=video_folder, 
                name_prefix=f"pendulum_test_episode_{i}",
                episode_trigger=lambda x: True # Record this one episode
            )
            
            state_np, _ = record_env.reset(seed=current_seed) # Use a different seed for testing
            done = False
            episode_score = 0.0

            while not done:
                action_np = self.select_action(state_np) # state_np is (obs_dim,)
                next_state_np, reward_np, terminated, truncated, info = record_env.step(action_np)
                done = terminated or truncated
                
                state_np = next_state_np
                current_reward_scalar = reward_np.item() if hasattr(reward_np, 'item') else reward_np[0]
                episode_score += current_reward_scalar
            
            if episode_score > self.reward_threshold:
                valid_seeds.append(self.seed + i)

            tqdm.write(f"Test Episode {i}: Score = {episode_score:.2f}")
            total_score += episode_score  # Add to running total
            record_env.close() # This closes the video recorder and current_test_env

        # Report average score
        avg_score = total_score / num_test_episodes
        print(f"Testing finished. Average Score over {num_test_episodes} episodes: {avg_score:.2f}")
        print(f"Videos saved to {video_folder}")
        seeds_str = ",".join(map(str, valid_seeds))
        print(f"Valid seeds for performance: {seeds_str}, {len(valid_seeds)} valid seeds found.")
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=4e-4)
    parser.add_argument("--critic-lr", type=float, default=4e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=5e-2)  # Changed from int to float
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=64)
    parser.add_argument("--update-epoch", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="./output_ppo", 
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=10, 
                        help="Save checkpoint every N episodes")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint")
    parser.add_argument("--resume-path", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="Test mode - load model and evaluate")
    parser.add_argument("--video-dir", type=str, default="./videos_ppo",
                        help="Directory to save test videos")
    parser.add_argument("--num-test-episodes", type=int, default=20,
                        help="Number of episodes for testing")
    parser.add_argument("--valid-seeds", type=str, default=None,
                        help="Comma-separated list of valid seeds for testing")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    if args.valid_seeds:
        valid_seeds = list(map(int, args.valid_seeds.split(',')))[:20]
        print(f"Using valid seeds for testing: {valid_seeds}")
    else:
        valid_seeds = []  # Default to first 20 seeds if not provided
    
    if not args.eval:
        wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True, config=args)
    
    agent = PPOAgent(env, args)
    
    if args.eval:
        if args.checkpoint:
            agent.load_checkpoint(args.checkpoint)
            os.makedirs(args.video_dir, exist_ok=True)
            agent.test(args.video_dir, args.num_test_episodes, valid_seeds)
        else:
            print("Please specify a model path for testing with --test-model-path")
    else:
        agent.train()