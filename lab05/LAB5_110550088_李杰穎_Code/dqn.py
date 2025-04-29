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
                nn.Conv2d(input_dim, 16, kernel_size=8, stride=4),  # (N, 32, 20, 20)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (N, 64, 9, 9)
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1),  # (N, 64, 7, 7)
                nn.ReLU(),
                nn.Flatten()  # (N, 64 * 7 * 7 = 3136)
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
        
        # # Normalize pixel values to 0-1 range
        # normalized = resized / 255.0
        
        # # Enhance contrast
        # normalized = np.clip((normalized - 0.2) * 1.5, 0, 1)
        
        # # Convert back to uint8 (0-255 range)
        # enhanced = (normalized * 255).astype(np.uint8)
        
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
    def __init__(self, capacity, alpha=0.5, beta=0.4):
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
        ########## YOUR CODE HERE (for Task 3) ########## 
        # Update the priorities of the sampled transitions
        for idx, error in zip(indices, errors):
            # Ensure idx is within valid range
            if idx < len(self.buffer):
                # Calculate new priority and update
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
                self.max_priority = max(self.max_priority, self.priorities[idx])

        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        if args.replay_buffer_type == "uniform":
            self.memory = UniformReplayBuffer(capacity=args.memory_size)
        elif args.replay_buffer_type == "prioritized":
            self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)
        else:
            raise ValueError("Invalid replay buffer type. Choose 'uniform' or 'prioritized'.")

        if env_name == "ALE/Pong-v5":
            self.preprocessor = AtariPreprocessor()
            self.input_dim = 4  # For convolutional input (4 stacked frames)
        elif env_name == "CartPole-v1":
            self.preprocessor = SimplePreprocessor()
            self.input_dim = self.env.observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported environment: {env_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.dueling = args.dueling_dqn

        self.q_net = DQN(self.input_dim, self.num_actions, conv=(env_name == "ALE/Pong-v5"), dueling=self.dueling).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.input_dim, self.num_actions, conv=(env_name == "ALE/Pong-v5"), dueling=self.dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_start = args.epsilon_start
        # self.epsilon_decay = args.epsilon_decay
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21 if env_name == "ALE/Pong-v5" else 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.train_frequency = args.train_frequency
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Double DQN
        self.double_dqn = args.double_dqn

        # Multi-step Q-learning
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Milestone
        self.milestone_steps = [200_000, 400_000, 600_000, 800_000, 1_000_000, 1_200_000, 2_000_000]
        self.saved_milestones = set()  # Keep track of which milestones we've saved

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
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
    

                # Use initial priority of 1.0 (high priority) for new experiences
                if self.env_count % self.train_frequency == 0:
                    for _ in range(self.train_per_step):
                        self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                # if self.env_count % 1000 == 0:
                #     if self.epsilon > self.epsilon_min:
                #         self.epsilon *= self.epsilon_decay
                #         self.epsilon = max(self.epsilon, self.epsilon_min)

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
              
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

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
        if len(self.memory) < self.replay_start_size:
            return 

        fraction = min(1.0, self.env_count / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_min - self.epsilon_start)
        self.train_count += 1

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        if states is None:
            return

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(1)[0]

            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        td_errors = q_values - target_q_values.detach()
        # use Huber loss
        loss = (weights * torch.nn.functional.smooth_l1_loss(q_values, target_q_values.detach(), reduction='none')).mean()
    
        # loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        with torch.no_grad():
            td_errors_np = td_errors.abs().cpu().numpy()
        self.memory.update_priorities(indices, td_errors_np)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

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
    parser.add_argument("--replay-buffer-type", type=str, default="uniform", choices=["uniform", "prioritized"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1000000)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=200)
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--train-frequency", type=int, default=4)
    # Double-DQN
    parser.add_argument("--double-dqn", action="store_true", default=False)
    # Multi-step Q-learning
    parser.add_argument("--n-step", type=int, default=1)
    # Dueling DQN
    parser.add_argument("--dueling-dqn", action="store_true", default=False, help="Use dueling network architecture")
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.num_episodes)
    wandb.finish()