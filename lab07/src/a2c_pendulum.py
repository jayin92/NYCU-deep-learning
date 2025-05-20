#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
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
from typing import Tuple
import os # Make sure os is imported
import time # Import time module for generating unique IDs

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

def initialize_weights(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def initialize_kaiming(layer: nn.Linear):
    """Initialize the weights and bias using Kaiming initialization."""
    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

def initialize_orthogonal(layer: nn.Linear, gain=1.0):
    """Initialize the weights and bias using orthogonal initialization."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, activation=nn.Tanh):
        """Initialize."""
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, out_dim)
        # self.logstds_layer = nn.Linear(hidden_dim, out_dim)
        self.activation = activation()
        self.logstds = nn.Parameter(torch.full((out_dim,), 0.5)) # Higher initial exploration

        # initialize_orthogonal(self.fc1)
        # initialize_orthogonal(self.fc2)
        # initialize_orthogonal(self.mean_layer, gain=0.01)        

        # initialize_weights(self.fc1)
        # initialize_weights(self.fc2)
        # initialize_weights(self.mean_layer)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation.
        Args:
            state (torch.Tensor): Input state, expected shape [batch_size, in_dim].
        Returns:
            Tuple[torch.Tensor, Normal]: Sampled action and the distribution. Action shape [batch_size, out_dim].
        """
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        means = self.mean_layer(x)
        # logstds = self.logstds_layer(x)
        
        stds = torch.clamp(torch.exp(self.logstds), 1e-5, 3.0) # Shape: [batch_size, out_dim]
        # print(f"Means: {means}, Stds: {stds}") # Debugging line
        
        dist = Normal(means, stds)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, activation=nn.Tanh):
        """Initialize."""
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Critic outputs a single value
        self.activation = activation()
        
        # initialize_orthogonal(self.fc1)
        # initialize_orthogonal(self.fc2)
        # initialize_orthogonal(self.fc3)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        Args:
            state (torch.Tensor): Input state, expected shape [batch_size, in_dim].
        Returns:
            torch.Tensor: State value, shape [batch_size, 1].
        """
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        value = self.fc3(x) # Shape: [batch_size, 1]
        return value
    

class A2CAgent:
    """A2CAgent interacting with environment."""

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.checkpoint_path = args.checkpoint # Use a consistent name
        self.eval_interval = args.eval_interval
        self.num_eval_episodes = args.num_eval_episodes
        self.video_dir = args.video_dir
        self.reward_threshold = args.reward_threshold
        self.use_mc_returns = args.use_mc_returns  # Use Monte Carlo returns instead of TD targets
        
        # Create a unique run ID for this training session to avoid overwriting videos
        self.run_id = int(time.time()) % 10000
        print(f"Run ID: {self.run_id}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] 
        
        # Select activation function
        activation = Mish if args.use_mish else nn.Tanh
        
        self.actor = Actor(self.obs_dim, self.action_dim, hidden_dim=args.hidden_dim, activation=activation).to(self.device)
        self.critic = Critic(self.obs_dim, hidden_dim=args.hidden_dim, activation=activation).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.transition: list = list()
        self.n_steps = args.n_steps
        self.total_step = 0
        self.is_test = False
        self.best_eval_score = float('-inf')  # Track best evaluation score

        self.normalize_rewards = args.normalize_rewards

    def select_action(self, state_np: np.ndarray) -> np.ndarray:
        """Select an action from the input state (NumPy array).
        Args:
            state_np (np.ndarray): Current state, can be (obs_dim,) or (1, obs_dim) or (obs_dim, 1).
        Returns:
            np.ndarray: Action to take, shape (action_dim,).
        """
        s_tensor = torch.tensor(state_np, dtype=torch.float32, device=self.device).reshape(1, self.obs_dim)

        action_tensor, dist = self.actor(s_tensor)
        selected_action_tensor = dist.mean if self.is_test else action_tensor

        # Apply tanh and scaling
        selected_action_tensor = torch.tanh(selected_action_tensor) * 2.0 # Scale to [-2, 2]
        # Clamp to ensure action is within bounds
        selected_action_tensor = torch.clamp(selected_action_tensor, -2.0, 2.0)

        selected_action_np = selected_action_tensor.cpu().detach().numpy()

        if not self.is_test:
            self.transition = [state_np, selected_action_np]
        
        action_to_env = selected_action_np.flatten()
        return action_to_env

    def step(self, selected_action_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Take an action and return the response of the env.
        Args:
            selected_action_np (np.ndarray): The action taken, shape (action_dim,).
        Returns:
            Tuple[np.ndarray, np.ndarray, bool]: next_state, reward, done
        """
        next_state_np, reward_np, terminated, truncated, _ = self.env.step(selected_action_np)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state_np, reward_np, done])

        return next_state_np, reward_np, done

    def update_model(self, states, actions, next_states, rewards, dones: torch.Tensor) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        # Reshape states to [1, obs_dim] for network input
        # Get current value and next state value
        assert len(states) == self.n_steps

        states = torch.cat(states, dim=0).view(-1, self.obs_dim) # Shape [N, obs_dim]
        actions = torch.cat(actions, dim=0).view(-1, self.action_dim) # Shape [N, action_dim]
        next_states = torch.cat(next_states, dim=0).view(-1, self.obs_dim) # Shape [N, obs_dim]
        rewards = torch.cat(rewards, dim=0).view(-1, 1) # Shape [N, 1]
        dones = torch.cat(dones, dim=0).view(-1, 1) # Shape [N, 1]

        if self.normalize_rewards:
            rewards = rewards / 20.0

        # Calculate returns based on strategy
        if self.use_mc_returns:
            # Monte Carlo returns (use if trajectory is complete)
            with torch.no_grad():
                R = self.critic(next_states[-1]) * (1 - dones[-1])  # Zero if terminal state

            returns = []
            for r, d in zip(reversed(rewards), reversed(dones)):
                R = r + self.gamma * R * (1 - d)
                returns.insert(0, R)
            td_targets = torch.cat(returns, dim=0)
        else:
            # TD targets (bootstrapping)
            with torch.no_grad():
                td_targets = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        values = self.critic(states)
        advantage = (td_targets - values).detach()

        # Actor
        _, dists = self.actor(states)
        log_probs = dists.log_prob(actions) # Shape [N, action_dim]
        entropies = dists.entropy() # Shape [N, action_dim]
        actor_loss = (-log_probs * advantage).mean() - self.entropy_weight * entropies.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Critic 
        value_loss = F.smooth_l1_loss(values, td_targets)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return actor_loss.item(), value_loss.item()

    def save_checkpoint(self, suffix=""):
        """Saves the model and optimizer states."""
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir): # Create dir if it doesn't exist
            os.makedirs(checkpoint_dir)
        
        # Add suffix if provided (e.g., for performance-based checkpoints)
        save_path = self.checkpoint_path
        if suffix:
            base, ext = os.path.splitext(self.checkpoint_path)
            save_path = f"{base}_{suffix}{ext}"
            
        print(f"Saving checkpoint to {save_path}")
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_step": self.total_step,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "best_eval_score": self.best_eval_score
        }, save_path)
    
    def load_checkpoint(self):
        """Loads the model and optimizer states."""
        if not os.path.exists(self.checkpoint_path):
            print(f"Checkpoint file not found: {self.checkpoint_path}")
            return
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Basic check for dimensionality consistency
        if "obs_dim" in checkpoint and checkpoint["obs_dim"] != self.obs_dim:
            print(f"Warning: Checkpoint obs_dim ({checkpoint['obs_dim']}) differs from current env obs_dim ({self.obs_dim}).")
        if "action_dim" in checkpoint and checkpoint["action_dim"] != self.action_dim:
             print(f"Warning: Checkpoint action_dim ({checkpoint['action_dim']}) differs from current env action_dim ({self.action_dim}).")

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_step = checkpoint.get("total_step", 0) # Load total_step, default to 0
        self.best_eval_score = checkpoint.get("best_eval_score", float('-inf'))
    
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
    
    def train(self):
        """Train the agent."""
        self.is_test = False
        self.actor.train() # Set actor to training mode
        self.critic.train() # Set critic to training mode
        # Buffers for n-step returns
        state_buffer = []
        action_buffer = []
        next_state_buffer = []
        rewards_buffer = []
        dones_buffer = []
        for ep in tqdm(range(1, self.num_episodes + 1), desc="Training Episodes"): 
            state_np, _ = self.env.reset(seed=self.seed + ep) # state_np shape: (obs_dim,) e.g. (3,)

            episode_score = 0.0 # Use float for score
            done = False
            episode_steps = 0
            while not done:                
                selected_action_np = self.select_action(state_np) # Returns 1D np.array (action_dim,)                
                next_state_np, reward_np, done = self.step(selected_action_np)
                state_np = next_state_np # state_np for the next iteration

                state, action, next_state, reward, done = self.transition

                # Convert to tensors
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, self.obs_dim)
                action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).reshape(1, self.action_dim)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).reshape(1, self.obs_dim)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(1, 1)
                done_tensor = torch.tensor(done, dtype=torch.float32, device=self.device).reshape(1, 1)

                state_buffer.append(state_tensor)
                action_buffer.append(action_tensor)
                next_state_buffer.append(next_state_tensor)
                rewards_buffer.append(reward_tensor)
                dones_buffer.append(done_tensor)
                
                if len(state_buffer) == self.n_steps:
                    actor_loss, critic_loss = self.update_model(state_buffer, action_buffer, next_state_buffer, rewards_buffer, dones_buffer)
                    wandb.log({
                        "train/step": self.total_step,
                        "train/actor_loss": actor_loss,
                        "train/critic_loss": critic_loss,
                    }) 
                    state_buffer.clear()
                    action_buffer.clear()
                    next_state_buffer.clear()
                    rewards_buffer.clear()
                    dones_buffer.clear()

                # reward_np from Pendulum is np.array([value]). Extract scalar.
                current_reward_scalar = float(reward_np)
                episode_score += current_reward_scalar
                
                self.total_step += 1
                episode_steps +=1
                
                if done:
                    tqdm.write(f"Episode {ep}: Total Reward = {episode_score:.2f}, Steps = {episode_steps}")
                    wandb.log({
                        "train/episode": ep,
                        "train/episodic_return": episode_score,
                        "train/episode_length": episode_steps
                    })

            # Regular checkpoint saving
            if ep % 50 == 0: # Save checkpoint every 50 episodes
                self.save_checkpoint()
                
            # Evaluation during training
            if ep % self.eval_interval == 0:
                eval_score = self.evaluate(ep)

        # Final checkpoint and evaluation
        self.save_checkpoint(suffix="final")
        self.evaluate(self.num_episodes)

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


def seed_torch(seed_value: int): # Added type hint
    torch.manual_seed(seed_value)
    random.seed(seed_value) # Seed python's random module
    np.random.seed(seed_value) # Seed numpy
    if torch.cuda.is_available(): # Seed cuda if available
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # for multi-GPU
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2C Agent for Pendulum-v1") # Added description
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab7-A2C-Pendulum", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run-improved", help="Weights & Biases run name") # Changed default
    parser.add_argument("--actor-lr", type=float, default=4e-4, help="Learning rate for the actor network")
    parser.add_argument("--critic-lr", type=float, default=4e-3, help="Learning rate for the critic network")
    parser.add_argument("--discount-factor", "--gamma", type=float, default=0.9, help="Discount factor for future rewards")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Total number of training episodes")
    parser.add_argument("--seed", type=int, default=77, help="Random seed for reproducibility")
    parser.add_argument("--entropy-weight", type=float, default=0.01, help="Weight for the entropy bonus (0 to disable)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Number of units in hidden layers")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save outputs (e.g., checkpoints)")
    parser.add_argument("--video-dir", type=str, default="./videos", help="Directory to save test videos") # Changed default
    parser.add_argument("--n-steps", type=int, default=32, help="Number of steps to look ahead")
    parser.add_argument("--checkpoint", type=str, default="./output/a2c_pendulum_checkpoint.pth", help="Path to save/load checkpoint") # Changed default
    parser.add_argument("--eval", action="store_true", default=False, help="Run in evaluation mode (load checkpoint and test)")
    parser.add_argument("--num-test-episodes", type=int, default=20, help="Number of episodes to run during testing")
    # New arguments for evaluation during training
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluate policy every N episodes during training")
    parser.add_argument("--num-eval-episodes", type=int, default=20, help="Number of episodes to run for each evaluation")
    parser.add_argument("--reward-threshold", type=float, default=-150, help="Reward threshold to consider task solved")
    # Training strategy arguments
    parser.add_argument("--use-mc-returns", action="store_true", default=False, help="Use Monte Carlo returns instead of TD targets")
    parser.add_argument("--use-mish", action="store_true", default=True, help="Use Mish activation function instead of tanh")
    # Normalized rewards
    parser.add_argument("--normalize-rewards", action="store_true", default=False, help="Normalize rewards")
    # Seeds
    parser.add_argument("--valid-seeds", type=str, default="", help="List of valid seeds for performance")

    args = parser.parse_args()

    args.valid_seeds = [int(seed) for seed in args.valid_seeds.split(",")] if args.valid_seeds else []
    args.valid_seeds = args.valid_seeds[:20]
    print(f"Found {len(args.valid_seeds)} valid seeds for performance: {args.valid_seeds}")
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    
    env = gym.make("Pendulum-v1", render_mode="rgb_array") # rgb_array for video recording
    
    seed_torch(args.seed)

    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args), save_code=True)
    
    agent = A2CAgent(env, args) # Pass args to agent

    if args.eval:
        agent.load_checkpoint()
        agent.test(video_folder=args.video_dir, num_test_episodes=args.num_test_episodes, seeds=args.valid_seeds) # Pass video_folder
    else:
        agent.train()

    env.close() # Close the main environment
    wandb.finish() # Ensure W&B run is finished