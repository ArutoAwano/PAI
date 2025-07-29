#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import yaml

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

def main():
    # データセットの設定
    dataset_root = "/home/ubuntu/dataset/lerobot_dataset/crane_plus_pekori"
    repo_id = "crane_plus_pekori"
    
    # データセットの読み込み
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")
    
    # アクションの統計を計算
    actions = []
    states = []
    
    for i in range(min(1000, len(dataset))):  # 最初の1000サンプルを分析
        sample = dataset[i]
        actions.append(sample["action"])
        states.append(sample["observation.state"])
    
    actions = np.array(actions)
    states = np.array(states)
    
    print("\n=== Action Statistics ===")
    print(f"Action shape: {actions.shape}")
    print(f"Action mean: {actions.mean(axis=0)}")
    print(f"Action std: {actions.std(axis=0)}")
    print(f"Action min: {actions.min(axis=0)}")
    print(f"Action max: {actions.max(axis=0)}")
    print(f"Action range: {actions.max(axis=0) - actions.min(axis=0)}")
    
    print("\n=== State Statistics ===")
    print(f"State shape: {states.shape}")
    print(f"State mean: {states.mean(axis=0)}")
    print(f"State std: {states.std(axis=0)}")
    print(f"State min: {states.min(axis=0)}")
    print(f"State max: {states.max(axis=0)}")
    
    print("\n=== Action Magnitude Analysis ===")
    action_magnitudes = np.linalg.norm(actions, axis=1)
    print(f"Action magnitude mean: {action_magnitudes.mean():.6f}")
    print(f"Action magnitude std: {action_magnitudes.std():.6f}")
    print(f"Action magnitude min: {action_magnitudes.min():.6f}")
    print(f"Action magnitude max: {action_magnitudes.max():.6f}")
    
    print("\n=== Individual Joint Analysis ===")
    joint_names = ["joint1", "joint2", "joint3", "joint4", "gripper"]
    for i, name in enumerate(joint_names):
        joint_actions = actions[:, i]
        print(f"{name}: mean={joint_actions.mean():.6f}, std={joint_actions.std():.6f}, range=[{joint_actions.min():.6f}, {joint_actions.max():.6f}]")
    
    # 大きなアクションの例を表示
    print("\n=== Large Action Examples ===")
    large_action_indices = np.where(action_magnitudes > action_magnitudes.mean() + 2 * action_magnitudes.std())[0]
    for i in large_action_indices[:5]:
        print(f"Sample {i}: action={actions[i]}, magnitude={action_magnitudes[i]:.6f}")

if __name__ == "__main__":
    main() 