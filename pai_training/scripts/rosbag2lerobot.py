#!/usr/bin/env python3

import numpy as np
import torch
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import yaml
import importlib.util
from typing import Tuple

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def filter_static_frames(joint_data: np.ndarray, gripper_data: np.ndarray, 
                        joint_threshold: float = 0.001, gripper_threshold: float = 0.001) -> np.ndarray:
    """静止フレームを除外するフィルター"""
    if len(joint_data) < 2:
        return np.ones(len(joint_data), dtype=bool)
    
    # 関節の変化量を計算
    joint_diff = np.abs(np.diff(joint_data, axis=0))
    joint_movement = np.any(joint_diff > joint_threshold, axis=1)
    
    # グリッパーの変化量を計算
    gripper_diff = np.abs(np.diff(gripper_data))
    gripper_movement = gripper_diff > gripper_threshold
    
    # 配列の長さを確認して調整
    min_length = min(len(joint_movement), len(gripper_movement))
    joint_movement = joint_movement[:min_length]
    gripper_movement = gripper_movement[:min_length]
    
    # どちらかが動いているフレームを保持
    movement_mask = np.logical_or(joint_movement, gripper_movement)
    
    # 最初と最後のフレームは常に保持
    keep_mask = np.ones(len(joint_data), dtype=bool)
    if len(movement_mask) > 0:
        keep_mask[1:1+len(movement_mask)] = movement_mask
    
    return keep_mask

def amplify_grip_frames(joint_data: np.ndarray, gripper_data: np.ndarray, 
                       action_data: np.ndarray, times: np.ndarray, 
                       amplification_factor: int = 50, window_size: int = 5,
                       noise_std: float = 0.01, action_noise_std: float = 0.005) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """グリップフレームを増幅する"""
    # グリッパーの状態変化を検出
    gripper_diff = np.diff(gripper_data)
    grip_points = np.where(np.abs(gripper_diff) > 0.01)[0]
    
    if len(grip_points) == 0:
        return joint_data, gripper_data, action_data, times
    
    # 増幅されたデータを格納するリスト
    amplified_joint = []
    amplified_gripper = []
    amplified_action = []
    amplified_times = []
    
    current_idx = 0
    for grip_point in grip_points:
        # グリップポイント前の通常フレームを追加
        start_idx = max(0, grip_point - window_size)
        if start_idx > current_idx:
            amplified_joint.extend(joint_data[current_idx:start_idx])
            amplified_gripper.extend(gripper_data[current_idx:start_idx])
            amplified_action.extend(action_data[current_idx:start_idx])
            amplified_times.extend(times[current_idx:start_idx])
        
        # グリップポイント周辺のフレームを増幅
        grip_start = max(0, grip_point - window_size)
        grip_end = min(len(joint_data), grip_point + window_size + 1)
        
        # 配列の長さを確認
        if grip_start >= grip_end:
            continue
            
        for _ in range(amplification_factor):
            # グリップフレーム周辺に小さなノイズを追加
            grip_joint = joint_data[grip_start:grip_end].copy()
            grip_gripper = gripper_data[grip_start:grip_end].copy()
            grip_action = action_data[grip_start:grip_end].copy()
            grip_times = times[grip_start:grip_end].copy()
            
            # グリップフレームに特別なノイズを追加
            grip_noise_std = noise_std * 2.0  # グリップ時は大きなノイズ
            grip_joint += np.random.normal(0, grip_noise_std, grip_joint.shape)
            grip_gripper += np.random.normal(0, grip_noise_std, grip_gripper.shape)
            grip_action += np.random.normal(0, action_noise_std * 2.0, grip_action.shape)
            
            amplified_joint.extend(grip_joint)
            amplified_gripper.extend(grip_gripper)
            amplified_action.extend(grip_action)
            amplified_times.extend(grip_times)
        
        current_idx = grip_end
    
    # 残りのフレームを追加
    if current_idx < len(joint_data):
        amplified_joint.extend(joint_data[current_idx:])
        amplified_gripper.extend(gripper_data[current_idx:])
        amplified_action.extend(action_data[current_idx:])
        amplified_times.extend(times[current_idx:])
    
    return (np.array(amplified_joint), np.array(amplified_gripper), 
            np.array(amplified_action), np.array(amplified_times))

class RosbagToLeRobot:
    def __init__(self, bag_dir, output_dir, config_path, target_freq=None, filter_static=True, 
                 joint_threshold=0.001, gripper_threshold=0.001, enable_grip_amplification=True,
                 grip_amplification_factor=50, grip_window_size=5):
        self.bag_path = Path(bag_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_freq = target_freq
        self.filter_static = filter_static
        self.joint_threshold = joint_threshold
        self.gripper_threshold = gripper_threshold
        self.enable_grip_amplification = enable_grip_amplification
        self.grip_amplification_factor = grip_amplification_factor
        self.grip_window_size = grip_window_size

        # Find all episode directories (containing .db3 and metadata.yaml)
        self.episode_dirs = sorted(p.parent for p in self.bag_path.rglob('*.db3'))
        if not self.episode_dirs:
            raise FileNotFoundError(f"No rosbag2 episode directories found in {bag_dir}")

        print(f"Found {len(self.episode_dirs)} episode directories")
        for ep_dir in self.episode_dirs:
            print(f"  - {ep_dir}")

        if config_path.endswith('.py'):
            config = load_config_py(config_path)
            features = config.features
            robot_type = getattr(config, "robot_type", "crane_plus")
            repo_id = getattr(config, "repo_id", "teleop_rosbag_dataset")
            task_name = getattr(config, "task_name", "teleop")
        else:
            # YAML fallback
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            features = config['features']
            robot_type = config.get('robot_type', 'crane_plus')
            repo_id = config.get('repo_id', 'teleop_rosbag_dataset')
            task_name = config.get('task_name', 'teleop')

        self.task_name = task_name

        # Create dataset
        try:
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=target_freq or 20,
                root=self.output_dir,
                robot_type=robot_type,
                features=features,
                use_videos=True,
            )
        except FileExistsError:
            print(f"[Info] Directory {self.output_dir} already exists. Loading existing dataset...")
            # 既存のデータセットを読み込み、新しいエピソードを追加
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=self.output_dir,
            )

        if "teleop" not in self.dataset.meta.task_to_task_index:
            self.dataset.meta.add_task("teleop")

    def _get_messages(self, episode_dir):
        """Get messages from all required topics in the rosbag."""
        messages = {
            '/joint_states': [],
            '/crane_plus_arm_controller/joint_trajectory': [],
            '/crane_plus_gripper_controller/joint_trajectory': [],
        }
        with Reader(episode_dir) as reader:
            connections = [x for x in reader.connections if x.topic in messages.keys()]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                messages[connection.topic].append((msg, timestamp))
        return messages

    def _sample_and_hold(self, times, values, target_times):
        values = np.asarray(values)
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
            new_values = np.zeros((len(target_times), values.shape[1]))
        elif len(values.shape) >= 3:
            new_values = np.zeros((len(target_times), *values.shape[1:]), dtype=values.dtype)
        else:
            new_values = np.zeros((len(target_times), values.shape[1]))
        for i, target_time in enumerate(target_times):
            idx = np.searchsorted(times, target_time, side='right') - 1
            if idx < 0:
                idx = 0
            new_values[i] = values[idx]
        if len(values.shape) == 1:
            new_values = new_values.flatten()
        return new_values

    def convert(self):
        max_frames = 0
        episode_lengths = []
        print("\nFirst pass: calculating episode lengths...")
        for episode_dir in self.episode_dirs:
            try:
                messages = self._get_messages(episode_dir)
                j_times = np.array([ts for _, ts in messages['/joint_states']])
                a_times = np.array([ts for _, ts in messages['/crane_plus_arm_controller/joint_trajectory']])
                g_times = np.array([ts for _, ts in messages['/crane_plus_gripper_controller/joint_trajectory']])
                # Use the intersection of available times
                start_time = max(j_times[0], a_times[0], g_times[0])
                end_time = min(j_times[-1], a_times[-1], g_times[-1])
                target_freq = self.target_freq or 20
                num_frames = int((end_time - start_time) * target_freq / 1e9)
                max_frames = max(max_frames, num_frames)
                episode_lengths.append(num_frames)
                print(f"Episode {episode_dir}: {num_frames} frames")
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
                episode_lengths.append(0)
                continue

        print(f"\nLongest episode: {max_frames} frames")

        for idx, episode_dir in enumerate(self.episode_dirs):
            print(f"\nProcessing episode {idx} from {episode_dir}...")
            try:
                messages = self._get_messages(episode_dir)
                joint_states = messages['/joint_states']
                arm_cmds = messages['/crane_plus_arm_controller/joint_trajectory']
                gripper_cmds = messages['/crane_plus_gripper_controller/joint_trajectory']
                j_times = np.array([ts for _, ts in joint_states])
                j_pos = np.array([msg.position for msg, _ in joint_states])
                a_times = np.array([ts for _, ts in arm_cmds])
                a_pos = np.array([msg.points[0].positions if msg.points else [0,0,0,0] for msg, _ in arm_cmds])
                g_times = np.array([ts for _, ts in gripper_cmds])
                g_pos = np.array([msg.points[0].positions[0] if (msg.points and len(msg.points[0].positions)>0) else 0.0 for msg, _ in gripper_cmds])
                start_time = max(j_times[0], a_times[0], g_times[0])
                end_time = min(j_times[-1], a_times[-1], g_times[-1])
                target_freq = self.target_freq or 20
                num_frames = episode_lengths[idx]
                target_times = np.linspace(start_time, end_time, num_frames)
                # Sample and hold for each
                j_pos_sampled = self._sample_and_hold(j_times, j_pos, target_times)  # (num_frames, 5)
                a_pos_sampled = self._sample_and_hold(a_times, a_pos, target_times)  # (num_frames, 4)
                g_pos_sampled = self._sample_and_hold(g_times, g_pos, target_times)  # (num_frames,)
                
                # 静止フレームを除外
                if self.filter_static:
                    keep_mask = filter_static_frames(j_pos_sampled, g_pos_sampled, 
                                                   self.joint_threshold, self.gripper_threshold)
                    j_pos_sampled = j_pos_sampled[keep_mask]
                    a_pos_sampled = a_pos_sampled[keep_mask]
                    g_pos_sampled = g_pos_sampled[keep_mask]
                    print(f"Static frame filtering: {len(keep_mask)} -> {keep_mask.sum()} frames")
                
                # グリップフレームを増幅
                if self.enable_grip_amplification:
                    j_pos_sampled, g_pos_sampled, a_pos_sampled, _ = amplify_grip_frames(
                        j_pos_sampled, g_pos_sampled, a_pos_sampled, np.zeros_like(g_pos_sampled),
                        self.grip_amplification_factor, self.grip_window_size,
                        noise_std=0.01, action_noise_std=0.005
                    )
                    print(f"Grip frame amplification: {len(j_pos_sampled)} frames")
                
                # Concatenate after sampling
                a_pos_full = np.hstack([a_pos_sampled, g_pos_sampled.reshape(-1,1)])
                
                for i in range(len(j_pos_sampled)):
                    self.dataset.add_frame({
                        "observation.state": torch.tensor(j_pos_sampled[i], dtype=torch.float32),
                        "observation.environment_state": torch.tensor(j_pos_sampled[i], dtype=torch.float32),
                        "action": torch.tensor(a_pos_full[i], dtype=torch.float32),
                        "task": self.task_name,
                    })
                print(f"Saving episode {idx}...")
                self.dataset.save_episode()
                print(f"Episode {idx} saved successfully")
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue
        return str(self.dataset.root)

def load_config_py(config_path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def main():
    # Load parameters from conversion_config.yaml
    with open("configs/conversion_config.yaml", "r") as f:
        conv_cfg = yaml.safe_load(f)
    bag_dir = conv_cfg["bag_dir"]
    output_dir = conv_cfg["output_dir"]
    config = conv_cfg["config"]
    target_freq = conv_cfg.get("target_freq", 20)
    
    # 静止フレーム除外の設定
    filter_static = conv_cfg.get("filter_static", True)
    joint_threshold = conv_cfg.get("joint_threshold", 0.001)
    gripper_threshold = conv_cfg.get("gripper_threshold", 0.001)

    # グリップ増幅の設定
    enable_grip_amplification = conv_cfg.get("enable_grip_amplification", True)
    grip_amplification_factor = conv_cfg.get("grip_amplification_factor", 50)
    grip_window_size = conv_cfg.get("grip_window_size", 5)

    converter = RosbagToLeRobot(bag_dir, output_dir, config, target_freq, 
                               filter_static, joint_threshold, gripper_threshold,
                               enable_grip_amplification, grip_amplification_factor, grip_window_size)
    output_path = converter.convert()
    print(f"Dataset saved to: {output_path}")

if __name__ == '__main__':
    main() 