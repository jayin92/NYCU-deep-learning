import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import itertools
import json
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import traceback
import sys
import copy

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
    DQN with optional dueling architecture
    """
    def __init__(self, input_dim, num_actions, conv=False, hidden_dim=64, dueling=False):
        super(DQN, self).__init__()
        self.conv = conv
        self.dueling = dueling
        self.num_actions = num_actions
        
        if conv:
            # Feature extraction layers for convolutional input
            self.features = nn.Sequential(
                nn.Conv2d(input_dim, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            feature_output_dim = 32 * 7 * 7
            
            if dueling:
                # Separate value and advantage streams for dueling architecture
                self.value_stream = nn.Sequential(
                    nn.Linear(feature_output_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )
                
                self.advantage_stream = nn.Sequential(
                    nn.Linear(feature_output_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, num_actions)
                )
            else:
                # Standard DQN output
                self.fc = nn.Sequential(
                    nn.Linear(feature_output_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions)
                )
        else:
            # Feature extraction for non-convolutional input
            self.features = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            if dueling:
                # Separate value and advantage streams for dueling architecture
                self.value_stream = nn.Sequential(
                    nn.Linear(hidden_dim, 1)
                )
                
                self.advantage_stream = nn.Sequential(
                    nn.Linear(hidden_dim, num_actions)
                )
            else:
                # Standard DQN output
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, num_actions)
                )

    def forward(self, x):
        if self.conv:
            x = x / 255.0
            
        features = self.features(x)
        
        if self.dueling:
            values = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Combine value and advantage streams using the dueling formula
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            return values + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            return self.fc(features)


class SimplePreprocessor:
    """
    Preprocessing the state input of DQN for CartPole
    """
    def __init__(self):
        pass

    def preprocess(self, obs):
        return obs

    def reset(self, obs):
        return self.preprocess(obs)

    def step(self, obs):
        return self.preprocess(obs)


class AtariPreprocessor:
    """
    Preprocessing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Crop irrelevant parts
        gray = gray[34:194, :]
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        _, thresholded = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        
        return thresholded

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.beta = 0.4  # Added for compatibility with PrioritizedReplayBuffer

    def add(self, transition, error):        
        # Add transition to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        # If buffer is empty or not enough samples, return None
        if len(self.buffer) == 0:
            return None, None, None, None, None, None, None
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
    
        indices = np.random.choice(len(self.buffer), batch_size)
        
        # Calculate importance sampling weights
        weights = np.ones((batch_size,), dtype=np.float32)
        
        # Get the sampled transitions
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, indices, errors):
        return


class PrioritizedReplayBuffer:
    """
    Prioritizing the samples in the replay memory by the Bellman error
    See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / 200000  # Increment beta gradually
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.pos = 0

    def add(self, transition, error=None):
        # Calculate priority based on TD error
        if error is None:
            priority = self.max_priority
        else:
            priority = (abs(error) + 1e-5) ** self.alpha
        
        # Add transition to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        # Update priority
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        # If buffer is empty or not enough samples, return None
        if len(self.buffer) == 0:
            return None, None, None, None, None, None, None
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices based on the probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Gradually increase beta to 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get the sampled transitions
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            # Ensure idx is within valid range
            if idx < len(self.buffer):
                # Calculate new priority and update
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
                self.max_priority = max(self.max_priority, self.priorities[idx])
        return


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", config=None):
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        
        # Store configuration
        self.config = config
        
        # Set up replay buffer
        if config.replay_buffer_type == "uniform":
            self.memory = UniformReplayBuffer(capacity=config.memory_size)
        elif config.replay_buffer_type == "prioritized":
            self.memory = PrioritizedReplayBuffer(capacity=config.memory_size, 
                                                 alpha=config.priority_alpha, 
                                                 beta=config.priority_beta)
        else:
            raise ValueError("Invalid replay buffer type. Choose 'uniform' or 'prioritized'.")

        # Set up state preprocessor
        if env_name == "ALE/Pong-v5":
            self.preprocessor = AtariPreprocessor()
            self.input_dim = 4  # For convolutional input (4 stacked frames)
        elif env_name == "CartPole-v1":
            self.preprocessor = SimplePreprocessor()
            self.input_dim = self.env.observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported environment: {env_name}")

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
        print(f"Using device: {self.device}")

        # Initialize networks
        self.dueling = config.dueling_dqn
        self.q_net = DQN(self.input_dim, self.num_actions, 
                         conv=(env_name == "ALE/Pong-v5"), 
                         hidden_dim=config.hidden_dim,
                         dueling=self.dueling).to(self.device)
        self.q_net.apply(init_weights)
        
        self.target_net = DQN(self.input_dim, self.num_actions, 
                              conv=(env_name == "ALE/Pong-v5"), 
                              hidden_dim=config.hidden_dim,
                              dueling=self.dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)

        # Store hyperparameters
        self.batch_size = config.batch_size
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_decay_steps = config.epsilon_decay_steps
        self.epsilon_min = config.epsilon_min

        # Tracking metrics
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21 if env_name == "ALE/Pong-v5" else 0
        self.eval_rewards = []
        
        # Additional parameters
        self.max_episode_steps = config.max_episode_steps
        self.replay_start_size = config.replay_start_size
        self.target_update_frequency = config.target_update_frequency
        self.train_per_step = config.train_per_step
        self.train_frequency = config.train_frequency
        self.save_dir = config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Advanced algorithms
        self.double_dqn = config.double_dqn
        self.n_step = config.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Evaluation results
        self.final_eval_reward = None
        
        # Benchmark metrics
        self.benchmark_step = 200000  # 200K steps benchmark
        self.benchmark_reward = None
        self.step_eval_rewards = {}  # Dictionary to store {step: reward} for benchmark comparisons

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000, eval_interval=20, eval_episodes=10):
        """Train the agent and evaluate periodically"""
        episode_rewards = []
        benchmark_done = False
        
        for ep in range(episodes):
            if benchmark_done:
                break
            obs, _ = self.env.reset()
            
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                
                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.n_step:
                    # Calculate n-step return
                    init_state, init_action, _, _, _ = self.n_step_buffer[0]
                    n_step_return = sum([r * (self.gamma ** i) for i, (_, _, r, _, _) in enumerate(self.n_step_buffer)])

                    final_next_state = self.n_step_buffer[-1][3]
                    final_done = self.n_step_buffer[-1][4]

                    self.memory.add((init_state, init_action, n_step_return, final_next_state, final_done), error=None)

                elif self.n_step == 1:
                    self.memory.add((state, action, reward, next_state, done), error=None)
    
                # Train if it's time
                if self.env_count % self.train_frequency == 0:
                    for _ in range(self.train_per_step):
                        self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                # Update epsilon based on fraction of decay steps
                fraction = min(1.0, self.env_count / self.epsilon_decay_steps)
                self.epsilon = self.epsilon_start + fraction * (self.epsilon_min - self.epsilon_start)

                # Check if we've reached benchmark step count (200K)
                if not benchmark_done and self.env_count >= self.benchmark_step:
                    print(f"\n[BENCHMARK - 200K STEPS] Running evaluation at {self.env_count} environment steps")
                    
                    # Perform benchmark evaluation
                    benchmark_rewards = []
                    for _ in range(20):  # Use more episodes for more stable benchmark
                        benchmark_reward = self.evaluate()
                        benchmark_rewards.append(benchmark_reward)
                    
                    self.benchmark_reward = sum(benchmark_rewards) / len(benchmark_rewards)
                    self.step_eval_rewards[self.env_count] = self.benchmark_reward
                    
                    print(f"[BENCHMARK RESULT] Reward at 200K steps: {self.benchmark_reward:.2f}")
                    wandb.log({
                        "Benchmark Step": self.env_count,
                        "Benchmark Reward": self.benchmark_reward
                    })
                    
                    benchmark_done = True
                    print(f"[BENCHMARK] Finished evaluation at {self.env_count} steps")
                    break

                # Log progress periodically
                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Memory Size": len(self.memory),
                        "Beta": self.memory.beta if hasattr(self.memory, 'beta') else 0.0
                    })
                
                # Also evaluate at specific step intervals (for tracking learning curve)
                if self.env_count % 50000 == 0:  # Every 50K steps
                    step_eval_rewards = []
                    for _ in range(10):
                        step_eval_reward = self.evaluate()
                        step_eval_rewards.append(step_eval_reward)
                    avg_step_eval_reward = sum(step_eval_rewards) / len(step_eval_rewards)
                    self.step_eval_rewards[self.env_count] = avg_step_eval_reward
                    
                    print(f"[Step Eval] Step: {self.env_count} Avg Eval Reward: {avg_step_eval_reward:.2f}")
                    wandb.log({
                        "Step": self.env_count,
                        "Step Eval Reward": avg_step_eval_reward
                    })
                      
            # End of episode logging
            episode_rewards.append(total_reward)
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Episode Length": step_count,
                "Average Reward": total_reward / max(1, step_count)
            })
            
            # Save model periodically
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            # Evaluate periodically
            if ep % eval_interval == 0:
                eval_rewards = []
                for _ in range(eval_episodes):
                    eval_reward = self.evaluate()
                    eval_rewards.append(eval_reward)
                avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
                self.eval_rewards.append(avg_eval_reward)
                
                if avg_eval_reward > self.best_reward:
                    self.best_reward = avg_eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {avg_eval_reward}")
                    
                print(f"[TrueEval] Ep: {ep} Avg Eval Reward: {avg_eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Eval Reward": avg_eval_reward
                })

            # Process transitions still in the buffer (less than n)
            while len(self.n_step_buffer) > 0:
                initial_state = self.n_step_buffer[0][0]
                initial_action = self.n_step_buffer[0][1]
                
                # Calculate return for remaining steps
                n_step_return = 0
                for i in range(len(self.n_step_buffer)):
                    n_step_return += (self.gamma**i) * self.n_step_buffer[i][2]
                
                final_next_state = self.n_step_buffer[-1][3]
                final_done = self.n_step_buffer[-1][4]
                
                # Add to buffer
                self.memory.add((initial_state, initial_action, n_step_return, final_next_state, final_done), error=None)
                
                # Remove the first item and continue
                self.n_step_buffer.popleft()
        
        # Make sure we did benchmark evaluation
        if not benchmark_done:
            print("\n[WARNING] Training finished before reaching 200K steps benchmark")
            # Run benchmark eval anyway if we have enough steps
            if self.env_count >= 0.8 * self.benchmark_step:  # If we've done at least 80% of benchmark steps
                benchmark_rewards = []
                for _ in range(20):
                    benchmark_reward = self.evaluate()
                    benchmark_rewards.append(benchmark_reward)
                
                self.benchmark_reward = sum(benchmark_rewards) / len(benchmark_rewards)
                print(f"[BENCHMARK RESULT] Reward at {self.env_count} steps (instead of 200K): {self.benchmark_reward:.2f}")
        
        # Final evaluation
        self.final_evaluation(eval_episodes=20)
        
        # Return both the final and benchmark rewards
        return self.benchmark_reward, self.final_eval_reward, self.step_eval_rewards
        
    def final_evaluation(self, eval_episodes=20):
        """Perform a final evaluation with more episodes"""
        eval_rewards = []
        for _ in range(eval_episodes):
            eval_reward = self.evaluate()
            eval_rewards.append(eval_reward)
        
        self.final_eval_reward = sum(eval_rewards) / len(eval_rewards)
        print(f"[Final Evaluation] Avg Reward: {self.final_eval_reward:.2f}")
        wandb.log({"Final Eval Reward": self.final_eval_reward})
        
        return self.final_eval_reward

    def evaluate(self):
        """Evaluate the agent without exploration"""
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward

    def train(self):
        """Train the agent on a batch of experience"""
        if len(self.memory) < self.replay_start_size:
            return 

        self.train_count += 1

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        if states is None:
            return

        # Convert batch to tensors
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Get current Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Select actions using policy network, evaluate with target network
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: directly use max Q-value from target network
                next_q_values = self.target_net(next_states).max(1)[0]

            # Calculate target Q values
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        # Calculate TD errors and loss
        td_errors = q_values - target_q_values.detach()
        loss = (weights * torch.nn.functional.smooth_l1_loss(q_values, target_q_values.detach(), reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        # Update priorities in replay buffer
        with torch.no_grad():
            td_errors_np = td_errors.abs().cpu().numpy()
        self.memory.update_priorities(indices, td_errors_np)

        # Update target network periodically
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Log training metrics periodically
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({
                "Training Loss": loss.item(),
                "Q Values Mean": q_values.mean().item(),
                "Q Values Std": q_values.std().item(),
                "Target Q Values Mean": target_q_values.mean().item()
            })


class Config:
    """Configuration class for hyperparameters"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)


def run_single_config(config_args):
    """
    Function to run a single hyperparameter configuration
    This will be executed in a separate process
    """
    try:
        config_id, config_dict, all_params, base_save_dir, eval_interval, env_name = config_args
        
        # Create a dedicated directory for this configuration
        config_save_dir = os.path.join(base_save_dir, config_id)
        os.makedirs(config_save_dir, exist_ok=True)
        
        # Add save_dir to parameters
        all_params["save_dir"] = config_save_dir
        
        # Save the configuration
        with open(os.path.join(config_save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
        
        # Create config object
        config = Config(**all_params)
        
        # Initialize WandB
        run_name = f"{env_name.replace('/', '-')}_{config_id}"
        wandb_run = wandb.init(
            project=f"DQN-GridSearch-{env_name.replace('/', '-')}",
            name=run_name,
            config=all_params,
            group="grid_search",
            reinit=True
        )
        
        print(f"\n[Process {os.getpid()}] Training configuration: {config_id}")
        
        # Train agent with this configuration
        agent = DQNAgent(env_name=env_name, config=config)
        benchmark_reward, final_reward, step_rewards = agent.run(
            episodes=all_params["episodes_per_config"],
            eval_interval=eval_interval
        )
        
        # Finish wandb run
        wandb_run.finish()
        
        # Return the results
        return {
            "config_id": config_id,
            "config": config_dict,
            "benchmark_reward": benchmark_reward,
            "final_eval_reward": final_reward,
            "step_eval_rewards": step_rewards,
            "best_eval_reward": agent.best_reward,
            "success": True
        }
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in process {os.getpid()} running config {config_id}:\n{error_msg}")
        return {
            "config_id": config_id,
            "config": config_dict if 'config_dict' in locals() else None,
            "error": str(e),
            "traceback": error_msg,
            "success": False
        }


def grid_search(env_name, param_grid, episodes_per_config=200, eval_interval=10, 
                base_save_dir="./grid_search_results", max_workers=8):
    """
    Perform parallel grid search over hyperparameter combinations
    
    Args:
        env_name: Name of the environment
        param_grid: Dictionary mapping parameter names to lists of values
        episodes_per_config: Number of episodes to train each configuration
        eval_interval: Evaluate every N episodes
        base_save_dir: Directory to save results
        max_workers: Maximum number of parallel processes to use
    
    Returns:
        best_config: The best hyperparameter configuration
        results: Dictionary mapping configuration to performance
    """
    # Create timestamp for this grid search run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_save_dir, f"grid_search_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Generate all combinations of hyperparameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Starting grid search with {len(param_combinations)} combinations using {max_workers} parallel workers")
    
    # Save parameter grid for reference
    with open(os.path.join(run_dir, "param_grid.json"), "w") as f:
        json.dump(param_grid, f, indent=4)
    
    # Create configuration arguments for each combination
    config_args_list = []
    for i, values in enumerate(param_combinations):
        config_dict = {name: value for name, value in zip(param_names, values)}
        config_id = f"config_{i:03d}"
        
        # Set fixed parameters
        all_params = {
            "env_name": env_name,
            "wandb_run_name": f"{env_name.replace('/', '-')}_{config_id}",
            "max_episode_steps": 1000000,
            "use_cuda": torch.cuda.is_available(),
            "episodes_per_config": episodes_per_config
        }
        
        # Update with grid search parameters
        all_params.update(config_dict)
        
        # Add to list of config arguments
        config_args_list.append((config_id, config_dict, all_params, run_dir, eval_interval, env_name))
    
    # Use multiprocessing to run configurations in parallel
    results = {}
    best_benchmark_reward = float('-inf')
    best_config = None
    best_config_id = None
    
    # Determine number of workers (capped by CPU count)
    num_workers = min(max_workers, len(config_args_list), multiprocessing.cpu_count())
    print(f"Using {num_workers} workers")
    
    # Run configurations in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_config = {executor.submit(run_single_config, config_args): config_args[0] for config_args in config_args_list}
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_config)):
            config_id = future_to_config[future]
            try:
                result = future.result()
                if result["success"]:
                    print(f"[{i+1}/{len(config_args_list)}] Completed {config_id}")
                    
                    # Store result
                    results[config_id] = result
                    
                    # Check if this is the best configuration so far
                    benchmark_reward = result.get("benchmark_reward")
                    if benchmark_reward is not None and benchmark_reward > best_benchmark_reward:
                        best_benchmark_reward = benchmark_reward
                        best_config = result["config"]
                        best_config_id = config_id
                        print(f"New best configuration: {config_id} with 200K benchmark reward {benchmark_reward:.2f}")
                else:
                    print(f"[{i+1}/{len(config_args_list)}] Failed {config_id}: {result.get('error', 'Unknown error')}")
                    results[config_id] = result
            except Exception as e:
                print(f"[{i+1}/{len(config_args_list)}] Failed {config_id} with exception: {e}")
                results[config_id] = {
                    "config_id": config_id,
                    "error": str(e),
                    "success": False
                }
            
            # Save results after each completion
            with open(os.path.join(run_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Final report
    print("\n" + "="*50)
    print("Grid Search Complete")
    print("="*50)
    
    if best_config is not None:
        print(f"Best configuration (based on 200K benchmark): {best_config_id}")
        print(f"Best 200K benchmark reward: {best_benchmark_reward:.2f}")
        print("Parameters:")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
        
        # Save best configuration
        with open(os.path.join(run_dir, "best_config.json"), "w") as f:
            json.dump({
                "config_id": best_config_id,
                "benchmark_reward": best_benchmark_reward,
                "parameters": best_config
            }, f, indent=4)
    else:
        print("No successful configurations found!")
    
    # Save all results in a more organized format for analysis
    benchmark_results = {}
    for config_id, result in results.items():
        if result.get("success", False):
            benchmark_results[config_id] = {
                "config": result["config"],
                "benchmark_reward": result.get("benchmark_reward", None),
            }
    
    with open(os.path.join(run_dir, "benchmark_results.json"), "w") as f:
        json.dump(benchmark_results, f, indent=4)
    
    return best_config, results


def train_best_model(env_name, best_config, episodes=1000, save_dir="./best_model"):
    """Train a model with the best configuration for more episodes"""
    print("\n" + "="*50)
    print("Training Best Model")
    print("="*50)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set fixed parameters
    all_params = {
        "env_name": env_name,
        "save_dir": save_dir,
        "wandb_run_name": f"{env_name}_best_model",
        "max_episode_steps": 1000,
        "replay_start_size": 1000,
        "train_frequency": 4,
        "train_per_step": 1,
        "use_cuda": torch.cuda.is_available(),
    }
    
    # Update with best parameters
    all_params.update(best_config)
    
    # Create config object
    config = Config(**all_params)
    
    # Initialize WandB
    wandb_run = wandb.init(
        project=f"DQN-BestModel-{env_name.replace('/', '-')}",
        name="best_model",
        config=all_params,
        reinit=True
    )
    
    # Train agent with best configuration
    agent = DQNAgent(env_name=env_name, config=config)
    benchmark_reward, final_reward, step_rewards = agent.run(episodes=episodes)
    
    print(f"Final evaluation reward: {final_reward:.2f}")
    if benchmark_reward is not None:
        print(f"Benchmark reward at 200K steps: {benchmark_reward:.2f}")
    
    # Create learning curve visualization
    if step_rewards:
        steps = sorted(step_rewards.keys())
        rewards = [step_rewards[step] for step in steps]
        
        # Log to wandb
        for step, reward in zip(steps, rewards):
            wandb.log({
                "Step": step,
                "Learning Curve Reward": reward
            })
        
        print("\nLearning curve:")
        for step, reward in zip(steps, rewards):
            print(f"  Step {step}: {reward:.2f}")
    
    # Finish wandb run
    wandb_run.finish()
    
    return benchmark_reward, final_reward


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Required for CUDA support in child processes
    
    parser = argparse.ArgumentParser(description="DQN with Parallel Grid Search")
    parser.add_argument("--env-name", type=str, default="CartPole-v1", choices=["CartPole-v1", "ALE/Pong-v5"])
    parser.add_argument("--grid-search", action="store_true", help="Perform grid search")
    parser.add_argument("--train-best", action="store_true", help="Train best model after grid search")
    parser.add_argument("--config-file", type=str, default=None, help="JSON file with configuration to use")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes for training best model")
    parser.add_argument("--episodes-per-config", type=int, default=200, help="Episodes per configuration during grid search")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reduced-grid", action="store_true", help="Use a reduced parameter grid for faster testing")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes to use")
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.grid_search:
        # Define hyperparameter grid
        if args.env_name == "CartPole-v1":
            if args.reduced_grid:
                # Reduced grid for faster testing
                param_grid = {
                    # Learning parameters
                    "lr": [0.001],
                    "batch_size": [64],
                    "discount_factor": [0.99],
                    
                    # Exploration parameters
                    "epsilon_start": [1.0],
                    "epsilon_min": [0.01],
                    "epsilon_decay_steps": [20000],
                    
                    # Network parameters
                    "hidden_dim": [128],
                    
                    # Update parameters
                    "target_update_frequency": [200],
                    
                    # Replay buffer parameters
                    "memory_size": [50000],
                    "replay_buffer_type": ["prioritized"],
                    "priority_alpha": [0.6],
                    "priority_beta": [0.4],
                    
                    # Algorithm variants
                    "double_dqn": [True],
                    "dueling_dqn": [True],
                    "n_step": [3]
                }
            else:
                # Full grid for thorough exploration
                param_grid = {
                    # Learning parameters
                    "lr": [0.0005, 0.001, 0.002],
                    "batch_size": [32, 64],
                    "discount_factor": [0.99],
                    
                    # Exploration parameters
                    "epsilon_start": [1.0],
                    "epsilon_min": [0.01, 0.05],
                    "epsilon_decay_steps": [10000, 20000, 50000],
                    
                    # Network parameters
                    "hidden_dim": [64, 128, 256],
                    
                    # Update parameters
                    "target_update_frequency": [100, 200, 500],
                    
                    # Replay buffer parameters
                    "memory_size": [10000, 50000],
                    "replay_buffer_type": ["uniform", "prioritized"],
                    "priority_alpha": [0.6],
                    "priority_beta": [0.4],
                    
                    # Algorithm variants
                    "double_dqn": [False, True],
                    "dueling_dqn": [False, True],
                    "n_step": [1, 3, 5]
                }
        else:  # Pong
            if args.reduced_grid:
                # Reduced grid for faster testing
                param_grid = {
                    "lr": [0.00025],
                    "batch_size": [32],
                    "discount_factor": [0.99],
                    "epsilon_start": [1.0],
                    "epsilon_min": [0.1],
                    "epsilon_decay_steps": [1000000],
                    "hidden_dim": [512],
                    "target_update_frequency": [10000],
                    "memory_size": [100000],
                    "replay_buffer_type": ["prioritized"],
                    "priority_alpha": [0.6],
                    "priority_beta": [0.4],
                    "double_dqn": [True],
                    "dueling_dqn": [True],
                    "n_step": [3]
                }
            else:
                # Full grid (still more constrained due to computational costs)
                print("Warning: Full grid search for Pong may take a long time!")
                param_grid = {
                    "lr": [0.0001, 0.00025],
                    "batch_size": [32, 16],
                    "discount_factor": [0.99],
                    "epsilon_start": [1.0],
                    "epsilon_min": [0.01],
                    "epsilon_decay_steps": [100000, 50000],
                    "hidden_dim": [64],
                    "target_update_frequency": [1000, 500],
                    "memory_size": [50000, 100000],
                    "replay_buffer_type": ["prioritized"],
                    "priority_alpha": [0.8, 0.7],
                    "priority_beta": [0.4],
                    "double_dqn": [True],
                    "dueling_dqn": [True],
                    "n_step": [3],
                    "replay_start_size": [5000],
                    "train_frequency": [4],
                    "train_per_step": [2, 4],
                }
            
        # Perform grid search with parallel processing
        best_config, results = grid_search(
            env_name=args.env_name,
            param_grid=param_grid,
            episodes_per_config=args.episodes_per_config,
            max_workers=args.workers
        )
        
        # Save best config to file for later use
        if best_config:
            with open("best_config.json", "w") as f:
                json.dump(best_config, f, indent=4)
                
            if args.train_best:
                # Train model with best configuration
                train_best_model(
                    env_name=args.env_name,
                    best_config=best_config,
                    episodes=args.episodes
                )
            
    elif args.config_file:
        # Load configuration from file
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
            
        # Add fixed parameters
        all_params = {
            "env_name": args.env_name,
            "save_dir": "./model_from_config",
            "wandb_run_name": f"{args.env_name}_from_config",
            "max_episode_steps": 1000,
            "replay_start_size": 1000,
            "train_frequency": 4,
            "train_per_step": 1,
            "use_cuda": torch.cuda.is_available(),
        }
        
        # Update with loaded parameters
        all_params.update(config_dict)
        
        # Create config object
        config = Config(**all_params)
        
        # Initialize WandB
        wandb_run = wandb.init(
            project=f"DQN-FromConfig-{args.env_name}",
            name="from_config",
            config=all_params
        )
        
        # Train agent with this configuration
        agent = DQNAgent(env_name=args.env_name, config=config)
        _, final_reward, _ = agent.run(episodes=args.episodes)
        
        print(f"Final evaluation reward: {final_reward:.2f}")
        if hasattr(agent, 'benchmark_reward') and agent.benchmark_reward is not None:
            print(f"Benchmark reward at 200K steps: {agent.benchmark_reward:.2f}")
        
        # Finish wandb run
        wandb_run.finish()
        
    else:
        # Default base configuration (if not doing grid search or loading from file)
        config_dict = {
            "replay_buffer_type": "prioritized",
            "batch_size": 64,
            "memory_size": 50000,
            "lr": 0.001,
            "discount_factor": 0.99,
            "epsilon_start": 1.0,
            "epsilon_decay_steps": 50000,
            "epsilon_min": 0.01,
            "target_update_frequency": 200,
            "double_dqn": True,
            "n_step": 3,
            "dueling_dqn": True,
            "hidden_dim": 128,
            "priority_alpha": 0.6,
            "priority_beta": 0.4
        }
        
        # Add fixed parameters
        all_params = {
            "env_name": args.env_name,
            "save_dir": "./default_model",
            "wandb_run_name": f"{args.env_name}_default",
            "max_episode_steps": 1000,
            "replay_start_size": 1000,
            "train_frequency": 4,
            "train_per_step": 1,
            "use_cuda": torch.cuda.is_available(),
        }
        
        # Update with default parameters
        all_params.update(config_dict)
        
        # Create config object
        config = Config(**all_params)
        
        # Initialize WandB
        wandb_run = wandb.init(
            project=f"DQN-Default-{args.env_name}",
            name="default_run",
            config=all_params
        )
        
        # Train agent with default configuration
        agent = DQNAgent(env_name=args.env_name, config=config)
        _, final_reward, _ = agent.run(episodes=args.episodes)
        
        print(f"Final evaluation reward: {final_reward:.2f}")
        if hasattr(agent, 'benchmark_reward') and agent.benchmark_reward is not None:
            print(f"Benchmark reward at 200K steps: {agent.benchmark_reward:.2f}")
        
        # Finish wandb run
        wandb_run.finish()