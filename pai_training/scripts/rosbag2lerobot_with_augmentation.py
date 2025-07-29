#!/usr/bin/env python3

import numpy as np
import torch
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import yaml
import importlib.util
import random
from typing import Dict, List, Tuple, Optional
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    print("Warning: scipy not available. Some augmentation features may not work.")
    interp1d = None
    gaussian_filter1d = None

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class DataAugmentation:
    """データ拡張クラス - cube graspingタスク用の様々な拡張手法を提供"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.noise_std = config.get('noise_std', 0.01)
        self.time_warp_factor = config.get('time_warp_factor', 0.1)
        self.action_noise_std = config.get('action_noise_std', 0.005)
        self.joint_limit_buffer = config.get('joint_limit_buffer', 0.05)
        
    def filter_static_frames(self, joint_data: np.ndarray, gripper_data: np.ndarray, 
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
        
    def add_gaussian_noise(self, data: np.ndarray, std: float = None) -> np.ndarray:
        """ガウシアンノイズを追加"""
        if std is None:
            std = self.noise_std
        return data + np.random.normal(0, std, data.shape)
    
    def time_warping(self, times: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """時間軸の歪曲による拡張"""
        if interp1d is None:
            print("Warning: scipy.interpolate not available, skipping time warping")
            return times, data
            
        # ランダムな時間歪曲を適用
        warp_factor = 1.0 + np.random.uniform(-self.time_warp_factor, self.time_warp_factor)
        warped_times = times * warp_factor
        
        # 新しい時間軸で補間
        f = interp1d(warped_times, data, axis=0, bounds_error=False, fill_value='extrapolate')
        new_times = np.linspace(warped_times[0], warped_times[-1], len(times))
        warped_data = f(new_times)
        
        return new_times, warped_data
    
    def action_smoothing(self, actions: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """アクションの平滑化"""
        if gaussian_filter1d is None:
            print("Warning: scipy.ndimage not available, skipping action smoothing")
            return actions
        return gaussian_filter1d(actions, sigma=sigma, axis=0)
    
    def joint_limit_aware_noise(self, joint_data: np.ndarray, limits: List[Tuple[float, float]]) -> np.ndarray:
        """関節制限を考慮したノイズ追加"""
        noisy_data = joint_data.copy()
        
        for i, (joint, (min_limit, max_limit)) in enumerate(zip(joint_data.T, limits)):
            # 制限に近い場合はノイズを小さくする
            margin = (max_limit - min_limit) * self.joint_limit_buffer
            safe_min = min_limit + margin
            safe_max = max_limit - margin
            
            # 安全範囲外の場合はノイズを制限
            mask = (joint < safe_min) | (joint > safe_max)
            noise_std = self.noise_std * 0.1 if mask.any() else self.noise_std
            
            noise = np.random.normal(0, noise_std, joint.shape)
            noisy_data[:, i] = joint + noise
            
            # 制限内にクリップ
            noisy_data[:, i] = np.clip(noisy_data[:, i], min_limit, max_limit)
        
        return noisy_data
    
    def trajectory_interpolation(self, data: np.ndarray, factor: int = 2) -> np.ndarray:
        """軌道の補間による拡張"""
        if interp1d is None:
            print("Warning: scipy.interpolate not available, skipping trajectory interpolation")
            return data
            
        if factor <= 1:
            return data
        
        # より細かい時間ステップで補間
        original_length = len(data)
        new_length = original_length * factor
        
        # 各次元で補間
        interpolated = np.zeros((new_length, data.shape[1]))
        for i in range(data.shape[1]):
            f = interp1d(np.arange(original_length), data[:, i], 
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
            interpolated[:, i] = f(np.linspace(0, original_length-1, new_length))
        
        return interpolated
    
    def grasp_phase_augmentation(self, joint_data: np.ndarray, gripper_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """グリッパーの開閉フェーズに基づく拡張"""
        # グリッパーの状態変化を検出
        gripper_diff = np.diff(gripper_data)
        grasp_points = np.where(np.abs(gripper_diff) > 0.01)[0]
        
        if len(grasp_points) > 0:
            # グラスプポイント周辺でノイズを調整
            for point in grasp_points:
                # グラスプ前後で異なるノイズレベル
                pre_grasp_noise = self.noise_std * 0.5  # グラスプ前は小さなノイズ
                post_grasp_noise = self.noise_std * 1.5  # グラスプ後は大きなノイズ
                
                # 適用範囲を設定
                pre_range = slice(max(0, point-10), point)
                post_range = slice(point, min(len(joint_data), point+10))
                
                if pre_range.start < pre_range.stop:
                    joint_data[pre_range] += np.random.normal(0, pre_grasp_noise, 
                                                             joint_data[pre_range].shape)
                if post_range.start < post_range.stop:
                    joint_data[post_range] += np.random.normal(0, post_grasp_noise, 
                                                              joint_data[post_range].shape)
        
        return joint_data, gripper_data
    
    def amplify_grip_frames(self, joint_data: np.ndarray, gripper_data: np.ndarray, 
                           action_data: np.ndarray, times: np.ndarray, 
                           amplification_factor: int = 50, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """グリップフレームを増幅する（ランダム性を抑えて元データに近づける）"""
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
                
            # 元のグリップフレームデータを取得
            original_grip_joint = joint_data[grip_start:grip_end]
            original_grip_gripper = gripper_data[grip_start:grip_end]
            original_grip_action = action_data[grip_start:grip_end]
            original_grip_times = times[grip_start:grip_end]
            
            for i in range(amplification_factor):
                # 元データをコピー（ノイズを最小限に）
                grip_joint = original_grip_joint.copy()
                grip_gripper = original_grip_gripper.copy()
                grip_action = original_grip_action.copy()
                grip_times = original_grip_times.copy()
                
                # 非常に小さなノイズのみ追加（元データを保持）
                # ノイズの標準偏差を大幅に削減
                grip_noise_std = self.noise_std * 0.1  # 元の1/10に削減
                action_noise_std = self.action_noise_std * 0.1  # 元の1/10に削減
                
                # グリップポイントに近いフレームほどノイズを小さくする
                for j in range(len(grip_joint)):
                    # グリップポイントからの距離に基づいてノイズを調整
                    distance_from_grip = abs(j - (grip_point - grip_start))
                    distance_factor = max(0.1, 1.0 - distance_from_grip / window_size)
                    
                    # 距離に応じてノイズを調整
                    adjusted_joint_noise = grip_noise_std * distance_factor * 0.5
                    adjusted_gripper_noise = grip_noise_std * distance_factor * 0.3
                    adjusted_action_noise = action_noise_std * distance_factor * 0.5
                    
                    # 最小限のノイズを追加
                    grip_joint[j] += np.random.normal(0, adjusted_joint_noise, grip_joint[j].shape)
                    grip_gripper[j] += np.random.normal(0, adjusted_gripper_noise)
                    grip_action[j] += np.random.normal(0, adjusted_action_noise, grip_action[j].shape)
                
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

class RosbagToLeRobotWithAugmentation:
    def __init__(self, bag_dir, output_dir, config_path, target_freq=None, augmentation_config=None):
        self.bag_path = Path(bag_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_freq = target_freq
        
        # データ拡張の設定
        if augmentation_config is None:
            augmentation_config = {
                'noise_std': 0.005,  # ノイズを削減
                'time_warp_factor': 0.05,  # 時間歪曲も削減
                'action_noise_std': 0.002,  # アクションノイズも削減
                'joint_limit_buffer': 0.05,
                'enable_augmentation': True,
                'augmentation_factor': 3,
                'joint_threshold': 0.001,
                'gripper_threshold': 0.001,
                'enable_grip_amplification': True,
                'grip_amplification_factor': 50,
                'grip_window_size': 5
            }
        self.augmentation = DataAugmentation(augmentation_config)
        self.augmentation_config = augmentation_config

        # Find all episode directories
        self.episode_dirs = sorted(p.parent for p in self.bag_path.rglob('*.db3'))
        if not self.episode_dirs:
            raise FileNotFoundError(f"No rosbag2 episode directories found in {bag_dir}")

        print(f"Found {len(self.episode_dirs)} episode directories")
        for ep_dir in self.episode_dirs:
            print(f"  - {ep_dir}")

        # Load configuration
        if config_path.endswith('.py'):
            config = load_config_py(config_path)
            features = config.features
            robot_type = getattr(config, "robot_type", "crane_plus")
            repo_id = getattr(config, "repo_id", "teleop_rosbag_dataset")
            task_name = getattr(config, "task_name", "teleop")
        else:
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

    def _apply_augmentation(self, j_pos_sampled, a_pos_sampled, g_pos_sampled, times):
        """データ拡張を適用"""
        if not self.augmentation_config.get('enable_augmentation', True):
            # 静止フレーム除外のみ適用
            keep_mask = self.augmentation.filter_static_frames(
                j_pos_sampled, g_pos_sampled,
                self.augmentation_config.get('joint_threshold', 0.001),
                self.augmentation_config.get('gripper_threshold', 0.001)
            )
            filtered_j_pos = j_pos_sampled[keep_mask]
            filtered_a_pos = a_pos_sampled[keep_mask]
            filtered_g_pos = g_pos_sampled[keep_mask]
            filtered_times = times[keep_mask]
            
            print(f"Static frame filtering: {len(j_pos_sampled)} -> {len(filtered_j_pos)} frames")
            
            # グリップフレーム増幅を適用
            if self.augmentation_config.get('enable_grip_amplification', True):
                amplification_factor = self.augmentation_config.get('grip_amplification_factor', 50)
                window_size = self.augmentation_config.get('grip_window_size', 5)
                
                amplified_j_pos, amplified_g_pos, amplified_a_pos, amplified_times = \
                    self.augmentation.amplify_grip_frames(
                        filtered_j_pos, filtered_g_pos, filtered_a_pos, filtered_times,
                        amplification_factor, window_size
                    )
                
                print(f"Grip frame amplification: {len(filtered_j_pos)} -> {len(amplified_j_pos)} frames")
                return [(amplified_j_pos, amplified_a_pos, amplified_g_pos, amplified_times)]
            
            return [(filtered_j_pos, filtered_a_pos, filtered_g_pos, filtered_times)]
        
        augmented_data = []
        augmentation_factor = self.augmentation_config.get('augmentation_factor', 3)
        
        # 元データを追加
        augmented_data.append((j_pos_sampled.copy(), a_pos_sampled.copy(), g_pos_sampled.copy(), times.copy()))
        
        # 拡張データを生成
        for i in range(augmentation_factor - 1):
            aug_j_pos = j_pos_sampled.copy()
            aug_a_pos = a_pos_sampled.copy()
            aug_g_pos = g_pos_sampled.copy()
            aug_times = times.copy()
            
            # 1. 時間歪曲
            if random.random() < 0.7:
                aug_times, aug_j_pos = self.augmentation.time_warping(aug_times, aug_j_pos)
                _, aug_a_pos = self.augmentation.time_warping(aug_times, aug_a_pos)
                _, aug_g_pos = self.augmentation.time_warping(aug_times, aug_g_pos.reshape(-1, 1))
                aug_g_pos = aug_g_pos.flatten()
            
            # 2. 関節制限を考慮したノイズ
            joint_limits = [
                (-np.pi, np.pi),  # joint1
                (-np.pi/2, np.pi/2),  # joint2
                (-np.pi, np.pi),  # joint3
                (-np.pi/2, np.pi/2),  # joint4
                (0, 1.0)  # gripper
            ]
            aug_j_pos = self.augmentation.joint_limit_aware_noise(aug_j_pos, joint_limits)
            
            # 3. アクションの平滑化
            if random.random() < 0.5:
                aug_a_pos = self.augmentation.action_smoothing(aug_a_pos, sigma=random.uniform(0.5, 1.5))
            
            # 4. グリッパーフェーズ拡張
            aug_j_pos, aug_g_pos = self.augmentation.grasp_phase_augmentation(aug_j_pos, aug_g_pos)
            
            # 5. アクションにノイズ追加
            aug_a_pos += np.random.normal(0, self.augmentation.action_noise_std, aug_a_pos.shape)
            
            # 6. 静止フレームを除外
            keep_mask = self.augmentation.filter_static_frames(
                aug_j_pos, aug_g_pos,
                self.augmentation_config.get('joint_threshold', 0.001),
                self.augmentation_config.get('gripper_threshold', 0.001)
            )
            aug_j_pos = aug_j_pos[keep_mask]
            aug_a_pos = aug_a_pos[keep_mask]
            aug_g_pos = aug_g_pos[keep_mask]
            aug_times = aug_times[keep_mask]
            
            # 7. グリップフレーム増幅
            if self.augmentation_config.get('enable_grip_amplification', True):
                amplification_factor = self.augmentation_config.get('grip_amplification_factor', 50)
                window_size = self.augmentation_config.get('grip_window_size', 5)
                
                aug_j_pos, aug_g_pos, aug_a_pos, aug_times = \
                    self.augmentation.amplify_grip_frames(
                        aug_j_pos, aug_g_pos, aug_a_pos, aug_times,
                        amplification_factor, window_size
                    )
            
            augmented_data.append((aug_j_pos, aug_a_pos, aug_g_pos, aug_times))
        
        return augmented_data

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
                j_pos_sampled = self._sample_and_hold(j_times, j_pos, target_times)
                a_pos_sampled = self._sample_and_hold(a_times, a_pos, target_times)
                g_pos_sampled = self._sample_and_hold(g_times, g_pos, target_times)
                
                # データ拡張を適用
                augmented_data = self._apply_augmentation(j_pos_sampled, a_pos_sampled, g_pos_sampled, target_times)
                
                # 各拡張データを保存
                for aug_idx, (aug_j_pos, aug_a_pos, aug_g_pos, aug_times) in enumerate(augmented_data):
                    # Concatenate after sampling
                    a_pos_full = np.hstack([aug_a_pos, aug_g_pos.reshape(-1,1)])
                    
                    # 静止フレームを除外
                    # keep_mask = self.augmentation.filter_static_frames(aug_j_pos, aug_g_pos) # This line is now handled by _apply_augmentation
                    # aug_j_pos = aug_j_pos[keep_mask]
                    # aug_a_pos = aug_a_pos[keep_mask]
                    # aug_g_pos = aug_g_pos[keep_mask]
                    # aug_times = aug_times[keep_mask]
                    
                    for i in range(max_frames):
                        idx_to_use = min(i, len(aug_j_pos) - 1)
                        self.dataset.add_frame({
                            "observation.state": torch.tensor(aug_j_pos[idx_to_use], dtype=torch.float32),
                            "observation.environment_state": torch.tensor(aug_j_pos[idx_to_use], dtype=torch.float32),
                            "action": torch.tensor(a_pos_full[idx_to_use], dtype=torch.float32),
                            "task": self.task_name,
                        })
                    
                    print(f"Saving episode {idx}_aug_{aug_idx}...")
                    self.dataset.save_episode()
                    print(f"Episode {idx}_aug_{aug_idx} saved successfully")
                    
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
    config_path = "configs/conversion_config.yaml"
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found. Please create it first.")
        return
        
    with open(config_path, "r") as f:
        conv_cfg = yaml.safe_load(f)
    
    bag_dir = conv_cfg["bag_dir"]
    output_dir = conv_cfg["output_dir"]
    config = conv_cfg["config"]
    target_freq = conv_cfg.get("target_freq", 20)
    
    # データ拡張の設定
    augmentation_config = {
        'noise_std': 0.005,  # ノイズを削減
        'time_warp_factor': 0.05,  # 時間歪曲も削減
        'action_noise_std': 0.002,  # アクションノイズも削減
        'joint_limit_buffer': 0.05,
        'enable_augmentation': True,
        'augmentation_factor': 3,
        'joint_threshold': 0.001,
        'gripper_threshold': 0.001,
        'enable_grip_amplification': True,
        'grip_amplification_factor': 50,
        'grip_window_size': 5
    }

    converter = RosbagToLeRobotWithAugmentation(bag_dir, output_dir, config, target_freq, augmentation_config)
    output_path = converter.convert()
    print(f"Dataset saved to: {output_path}")

if __name__ == '__main__':
    main() 