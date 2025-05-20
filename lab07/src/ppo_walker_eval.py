#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Script to load, evaluate and record Walker2d PPO agent

import os
import argparse
import gymnasium as gym
import torch
import numpy as np
import random
from ppo_walker import PPOAgent, seed_torch  # Assuming the provided code is saved as ppo_walker.py

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO agent and record videos")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved model checkpoint")
    parser.add_argument("--video-dir", type=str, default="./videos",
                        help="Directory to save recorded videos")
    parser.add_argument("--num-episodes", type=int, default=3,
                        help="Number of episodes to evaluate and record")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        help="Render mode for the environment")
    return parser.parse_args()

def setup_environment():
    args = parse_args()
    
    # Set up random seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    # Create video directory if it doesn't exist
    os.makedirs(args.video_dir, exist_ok=True)
    
    # Create environment with rendering enabled
    env = gym.make("Walker2d-v4", render_mode=args.render_mode)
    
    return env, args

def evaluate_and_record(env, args):
    # Configure agent parameters
    agent_args = argparse.Namespace(
        discount_factor=0.99,
        tau=0.95,
        batch_size=32,
        epsilon=0.1,
        update_epoch=20,
        rollout_len=512,
        entropy_weight=1e-2,
        actor_lr=5e-5,
        critic_lr=2e-4,
        seed=args.seed,
        normalize_obs=False,
        normalize_reward=False,
        checkpoint_dir="./output_walker",
        checkpoint_freq=10,
        num_test_episodes=args.num_episodes,
        milestones=[1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]
    )
    
    # Initialize agent
    agent = PPOAgent(env, agent_args)
    
    # Load checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    if not agent.load_checkpoint(args.checkpoint):
        print("Failed to load checkpoint. Exiting.")
        return
    
    print(f"Evaluating agent for {args.num_episodes} episodes and recording videos to {args.video_dir}")
    
    # Modified evaluation loop with video recording
    agent.is_test = True
    agent.actor.eval()
    agent.critic.eval()
    
    total_score = 0.0
    
    for i in range(args.num_episodes):
        # Create a new environment instance for recording
        test_env = gym.make("Walker2d-v4", render_mode=args.render_mode)
        
        # Wrap the environment with RecordVideo
        record_env = gym.wrappers.RecordVideo(
            test_env, 
            video_folder=args.video_dir, 
            name_prefix=f"walker_episode_{i}",
            episode_trigger=lambda x: True  # Record every episode
        )
        
        # Run a single episode
        current_seed = agent.seed + i
        state, _ = record_env.reset(seed=current_seed)
        done = False
        episode_score = 0.0
        
        print(f"Recording episode {i+1}/{args.num_episodes} with seed {current_seed}")
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = record_env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_score += reward
        
        print(f"Episode {i+1} score: {episode_score:.2f}")
        total_score += episode_score
        
        # Close the environment to ensure video is saved
        record_env.close()
    
    avg_score = total_score / args.num_episodes
    print(f"\nEvaluation complete!")
    print(f"Average score over {args.num_episodes} episodes: {avg_score:.2f}")
    print(f"Videos saved to: {args.video_dir}")

if __name__ == "__main__":
    env, args = setup_environment()
    evaluate_and_record(env, args)