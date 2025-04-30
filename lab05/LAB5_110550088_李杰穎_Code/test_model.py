import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
import argparse
from collections import deque

gym.register_envs(ale_py)


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""
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
        # Generate Gaussian noise and apply scaling transformation
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


class VanillaDQN(nn.Module):
    def __init__(self, input_dim, num_actions, conv=False, hidden_dim=512, dueling=False, noisy=False, vanilla=False):
        super(VanillaDQN, self).__init__()
        self.conv = conv
        self.dueling = dueling
        self.noisy = noisy
        self.num_actions = num_actions
        
        # Choose the appropriate linear layer based on noisy flag
        LinearLayer = NoisyLinear if noisy else nn.Linear
        self.multiplier = 2 if vanilla else 1
        if conv:
            self.network = nn.Sequential(
                nn.Conv2d(input_dim, 16 * self.multiplier, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16 * self.multiplier, 32 * self.multiplier, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32 * self.multiplier, 32 * self.multiplier, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                LinearLayer(32 * 7 * 7 * self.multiplier, 512),
                nn.ReLU(),
                LinearLayer(512, num_actions)
            )
        else:
            self.network = nn.Sequential(
                LinearLayer(input_dim, hidden_dim),
                nn.ReLU(),
                LinearLayer(hidden_dim, hidden_dim),
                nn.ReLU(),
                LinearLayer(hidden_dim, num_actions)
            )

    def forward(self, x):
        if self.conv:
            x = x / 255.0
            
        return self.network(x)


class DQN(nn.Module):
    """DQN with optional dueling architecture and noisy networks"""
    def __init__(self, input_dim, num_actions, conv=False, hidden_dim=64, dueling=False, noisy=False, vanilla=False):
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
    """Preprocessing the state input for CartPole"""
    def __init__(self):
        pass

    def preprocess(self, obs):
        return obs

    def reset(self, obs):
        return self.preprocess(obs)

    def step(self, obs):
        return self.preprocess(obs)


class AtariPreprocessor:
    """Preprocessing the state input for Atari"""
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

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    num_actions = env.action_space.n
    
    # Set up preprocessor based on environment
    if args.env_name == "ALE/Pong-v5":
        preprocessor = AtariPreprocessor() if not args.vanilla else VanillaAtariPreprocessor()
        input_dim = 4  # For convolutional input (4 stacked frames)
        conv = True
    elif args.env_name == "CartPole-v1":
        preprocessor = SimplePreprocessor()
        input_dim = env.observation_space.shape[0]
        conv = False
    else:
        raise ValueError(f"Unsupported environment: {args.env_name}")

    # Create model with the same architecture as used in training
    if args.vanilla:
        model = VanillaDQN(
            input_dim=input_dim,
            num_actions=num_actions,
            conv=conv,
            hidden_dim=args.hidden_dim,
            dueling=args.dueling_dqn,
            noisy=args.noisy_net,
            vanilla=args.vanilla
        ).to(device)
    else:
        model = DQN(
            input_dim=input_dim,
            num_actions=num_actions,
            conv=conv,
            hidden_dim=args.hidden_dim,
            dueling=args.dueling_dqn,
            noisy=args.noisy_net,
            vanilla=args.vanilla
        ).to(device)
    
    # Load model parameters
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluation episodes
    all_rewards = []
    for ep in range(args.episodes):
        # Reset noise if using noisy networks
        if args.noisy_net:
            model.reset_noise()
            
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        step_count = 0

        while not done and step_count < args.max_steps:
            # Render frame
            frame = env.render()
            frames.append(frame)

            # Select action
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            # Execute action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
            step_count += 1

        # Save video
        out_path = os.path.join(args.output_dir, f"{args.env_name.replace('/', '_')}_ep{ep}_reward{total_reward}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        
        all_rewards.append(total_reward)
        print(f"Episode {ep}: Total reward = {total_reward}, Steps = {step_count}, Video saved to {out_path}")

    # Report average performance
    avg_reward = sum(all_rewards) / len(all_rewards)
    print(f"\nEvaluation completed: Average reward over {args.episodes} episodes: {avg_reward:.2f}")
    print(f"Individual episode rewards: {all_rewards}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", choices=["CartPole-v1", "ALE/Pong-v5"], 
                       help="Environment to evaluate on")
    parser.add_argument("--output-dir", type=str, default="./eval_videos", help="Directory to save evaluation videos")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    
    # Model architecture parameters (must match the trained model)
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--dueling-dqn", action="store_true", default=False, help="Use dueling DQN architecture")
    parser.add_argument("--noisy-net", action="store_true", default=False, help="Use noisy networks")
    parser.add_argument("--vanilla", action="store_true", default=False, help="Use vanilla networks")
    
    args = parser.parse_args()
    evaluate(args)