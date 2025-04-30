import torch
import torch.nn as nn
import torch.nn.functional as F
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

gym.register_envs(ale_py)


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration
    Based on the paper: Noisy Networks for Exploration (Fortunato et al., 2018)
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Register buffers for noise
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorized Gaussian noise (outer product)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        # Generate Gaussian noise and apply scaling transformation f(x) = sgn(x) * sqrt(|x|)
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            # During training, use noisy weights
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation, use just the expected weights
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # Skip NoisyLinear layers as they have their own initialization


class DQN(nn.Module):
    """
    DQN with optional dueling architecture and noisy networks
    """
    def __init__(self, input_dim, num_actions, conv=False, hidden_dim=64, dueling=False, noisy=False):
        super(DQN, self).__init__()
        self.conv = conv
        self.dueling = dueling
        self.noisy = noisy
        self.num_actions = num_actions
        
        # Choose the appropriate linear layer based on noisy flag
        LinearLayer = NoisyLinear if noisy else nn.Linear
        
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
                    LinearLayer(feature_output_dim, 256),
                    nn.ReLU(),
                    LinearLayer(256, 1)
                )
                
                self.advantage_stream = nn.Sequential(
                    LinearLayer(feature_output_dim, 256),
                    nn.ReLU(),
                    LinearLayer(256, num_actions)
                )
            else:
                # Standard DQN output
                self.fc = nn.Sequential(
                    LinearLayer(feature_output_dim, 512),
                    nn.ReLU(),
                    LinearLayer(512, num_actions)
                )
        else:
            # Feature extraction for non-convolutional input
            self.features = nn.Sequential(
                LinearLayer(input_dim, hidden_dim),
                nn.ReLU(),
                LinearLayer(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            if dueling:
                # Separate value and advantage streams for dueling architecture
                self.value_stream = nn.Sequential(
                    LinearLayer(hidden_dim, 1)
                )
                
                self.advantage_stream = nn.Sequential(
                    LinearLayer(hidden_dim, num_actions)
                )
            else:
                # Standard DQN output
                self.fc = nn.Sequential(
                    LinearLayer(hidden_dim, num_actions)
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
    
    def reset_noise(self):
        """Reset the noise in all noisy layers"""
        if not self.noisy:
            return
        
        # Reset noise in all modules that are NoisyLinear
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


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
        Preprocesing the state input of DQN for Atari
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

class VanillaAtariPreprocessor:
    """Preprocessing the state input for Atari"""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class UniformReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.beta = beta
        self.buffer = []
        self.pos = 0

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
            return None, None, None, None, None, None
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
    def __init__(self, capacity, alpha=0.7, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / 200_000  # Increment beta gradually
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
            return None, None, None, None, None, None
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
        # Update the priorities of the sampled transitions
        for idx, error in zip(indices, errors):
            # Ensure idx is within valid range
            if idx < len(self.buffer):
                # Calculate new priority and update
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
                self.max_priority = max(self.max_priority, self.priorities[idx])
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        
        # Set up replay buffer
        if args.replay_buffer_type == "uniform":
            self.memory = UniformReplayBuffer(capacity=args.memory_size)
        elif args.replay_buffer_type == "prioritized":
            self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)
        else:
            raise ValueError("Invalid replay buffer type. Choose 'uniform' or 'prioritized'.")

        # Set up preprocessor based on environment
        if env_name == "ALE/Pong-v5":
            if args.vanilla_atari_preprocessor:
                self.preprocessor = VanillaAtariPreprocessor()
            else:
                self.preprocessor = AtariPreprocessor()
            self.input_dim = 4  # For convolutional input (4 stacked frames)
        elif env_name == "CartPole-v1":
            self.preprocessor = SimplePreprocessor()
            self.input_dim = self.env.observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported environment: {env_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Store architecture flags
        self.dueling = args.dueling_dqn
        self.noisy = args.noisy_net

        # Initialize networks
        self.q_net = DQN(
            self.input_dim, 
            self.num_actions, 
            conv=(env_name == "ALE/Pong-v5"), 
            hidden_dim=args.hidden_dim,
            dueling=self.dueling, 
            noisy=self.noisy
        ).to(self.device)
        
        self.q_net.apply(init_weights)
        
        self.target_net = DQN(
            self.input_dim, 
            self.num_actions, 
            conv=(env_name == "ALE/Pong-v5"), 
            hidden_dim=args.hidden_dim,
            dueling=self.dueling, 
            noisy=self.noisy
        ).to(self.device)
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        if self.noisy:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)

        # Hyperparameters
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_start = args.epsilon_start
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_min = args.epsilon_min

        # Training counters
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21 if env_name == "ALE/Pong-v5" else 0
        
        # Training settings
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.train_frequency = args.train_frequency
        self.mse_loss = args.mse_loss
        
        # Saving settings
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Algorithm variants
        self.double_dqn = args.double_dqn
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Milestone tracking
        self.milestone_steps = [200_000, 400_000, 600_000, 800_000, 1_000_000, 1_200_000, 2_000_000]
        self.saved_milestones = set()  # Keep track of which milestones we've saved


    def select_action(self, state):
        # With noisy networks, we can optionally reduce or eliminate ε-greedy exploration
        if self.noisy and not args.use_epsilon_with_noisy:
            # When using noisy nets, exploration is handled by the network noise
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()
        else:
            # Standard epsilon-greedy exploration
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            # Reset noise at the beginning of each episode
            if self.noisy:
                self.q_net.reset_noise()
                
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
    
                # Train the network
                if self.env_count % self.train_frequency == 0:
                    for _ in range(self.train_per_step):
                        self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                # Update epsilon if using epsilon-greedy (potentially alongside noisy nets)
                if not self.noisy or args.use_epsilon_with_noisy:
                    fraction = min(1.0, self.env_count / self.epsilon_decay_steps)
                    self.epsilon = self.epsilon_start + fraction * (self.epsilon_min - self.epsilon_start)

                # Check for milestone steps
                for milestone in self.milestone_steps:
                    if self.env_count >= milestone and milestone not in self.saved_milestones:
                        model_path = os.path.join(self.save_dir, f"model_step{milestone}.pt")
                        torch.save(self.q_net.state_dict(), model_path)
                        print(f"[Milestone] Saved checkpoint at {milestone} environment steps to {model_path}")
                        wandb.log({
                            "Milestone": milestone,
                            "Checkpoint Saved": True
                        })
                        self.saved_milestones.add(milestone)
                        
                        # Evaluate at milestone
                        eval_rewards = []
                        for _ in range(10):
                            eval_reward = self.evaluate()
                            eval_rewards.append(eval_reward)
                        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
                        print(f"[Milestone Eval] Step: {milestone} Avg Eval Reward: {avg_eval_reward:.2f}")
                        wandb.log({
                            "Milestone": milestone,
                            "Milestone Eval Reward": avg_eval_reward
                        })

                # Periodic logging
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    # Additional wandb logs for debugging
                    wandb.log({
                        "Memory Size": len(self.memory),
                        "Beta": self.memory.beta
                    })
                      
            # End of episode logging
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
            })
            # Additional end-of-episode logging
            wandb.log({
                "Episode Length": step_count,
                "Average Reward": total_reward / max(1, step_count)
            })
              
            # Periodic model saving
            if ep % 20 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            # Periodic evaluation
            if ep % 20 == 0:
                eval_rewards = []
                for _ in range(10):
                    eval_reward = self.evaluate()
                    eval_rewards.append(eval_reward)
                avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
                if avg_eval_reward > self.best_reward:
                    self.best_reward = avg_eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {avg_eval_reward}")
                print(f"[TrueEval] Ep: {ep} Avg Eval Reward: {avg_eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
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
        
    def evaluate(self):
        """Evaluate performance without exploration"""
        # Set the network to eval mode to disable noise for consistent evaluation
        self.q_net.eval()
        
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
        
        # Set the network back to training mode
        self.q_net.train()
        
        return total_reward

    def train(self):
        """Train the network on a batch of experience"""
        if len(self.memory) < self.replay_start_size:
            return 

        # Update epsilon if not using noisy networks for exploration
        if not self.noisy or args.use_epsilon_with_noisy:
            fraction = min(1.0, self.env_count / self.epsilon_decay_steps)
            self.epsilon = self.epsilon_start + fraction * (self.epsilon_min - self.epsilon_start)
        
        self.train_count += 1

        # Sample from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        if states is None:
            return

        # Convert to tensors
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
                # Double DQN: Select actions using online network, evaluate with target network
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: Use maximum Q-value from target network
                next_q_values = self.target_net(next_states).max(1)[0]

            # Calculate target Q values with n-step returns
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        # Calculate TD errors for prioritized replay
        td_errors = q_values - target_q_values.detach()
        
        if self.mse_loss:
            # Use MSE loss for simplicity
            loss = (weights * (td_errors ** 2)).mean()
        else:
            # Use Huber loss for stability
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

        # Update target network
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
            # Reset noise in target network if using noisy networks
            if self.noisy:
                self.target_net.reset_noise()

        # Periodic logging
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({
                "Training Loss": loss.item(),
                "Q Values Mean": q_values.mean().item(),
                "Q Values Std": q_values.std().item(),
                "Target Q Values Mean": target_q_values.mean().item()
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="CartPole-v1", choices=["CartPole-v1", "ALE/Pong-v5"])   
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--replay-buffer-type", type=str, default="prioritized", choices=["uniform", "prioritized"])
    parser.add_argument("--batch-size", type=int, default=128)  # Increased batch size
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.0005)  # Lower learning rate
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20000)  # Faster epsilon decay
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=100)  # More frequent target updates
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--train-per-step", type=int, default=4)  # More updates per step
    parser.add_argument("--train-frequency", type=int, default=1)  # Train every step
    parser.add_argument("--hidden-dim", type=int, default=256)  # Larger network
    
    # Algorithm variants
    parser.add_argument("--double-dqn", action="store_true", default=True)  # Enable by default
    parser.add_argument("--n-step", type=int, default=3)  # Use n-step returns
    parser.add_argument("--dueling-dqn", action="store_true", default=True)  # Enable by default
    
    # Noisy Networks
    parser.add_argument("--noisy-net", action="store_true", default=False, help="Use noisy networks for exploration")
    parser.add_argument("--use-epsilon-with-noisy", action="store_true", default=False, 
                      help="Use epsilon-greedy along with noisy networks (usually not needed)")
    
    # Ablate on MSE
    parser.add_argument("--mse-loss", action="store_true", default=False, help="Use MSE loss instead of Huber loss")

    # Vanilla Atari Preprocessor
    parser.add_argument("--vanilla-atari-preprocessor", action="store_true", default=False, 
                      help="Use vanilla Atari preprocessor (no cropping or resizing)")

    
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize WandB
    wandb.init(
        project="DLP-Lab5-DQN-CartPole", 
        name=args.wandb_run_name, 
        save_code=True,
        config=vars(args)
    )
    
    # Create and run agent
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.num_episodes)
    wandb.finish()