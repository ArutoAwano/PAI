#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import argparse
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional

from pai_training.models.cube_grasp_policy import create_cube_grasp_policy

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'

class CubeGraspInferenceNode(Node):
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        super().__init__('crane_plus_cube_grasp_infer')
        
        # パブリッシャーとサブスクライバーの設定
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        # モデルの読み込み
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.policy = self.load_model()
        self.policy.eval()
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        
        # 状態管理
        self.last_action = np.zeros(5, dtype=np.float32)
        self.latest_joint_state = None
        self.joint_history = []  # 時系列データ用
        self.max_history_length = 32  # 履歴の最大長
        
        # タイマー設定（20Hz）
        self.timer = self.create_timer(0.05, self.publish_action)
        
        # ログ設定
        self.get_logger().info(f"CubeGraspPolicy inference node initialized")
        self.get_logger().info(f"Model loaded from: {model_path}")
        self.get_logger().info(f"Device: {self.device}")
        
    def load_model(self):
        """モデルの読み込み"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # チェックポイントの読み込み
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # 設定の取得
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # デフォルト設定
            config = {
                'model_config': {
                    'policy_type': 'standard',
                    'state_dim': 5,
                    'action_dim': 5,
                    'hidden_dim': 256,
                    'num_layers': 4,
                    'dropout': 0.1,
                    'use_grasp_attention': True,
                    'use_phase_detection': True
                }
            }
        
        # モデルの作成
        model = create_cube_grasp_policy(config['model_config'])
        
        # 重みの読み込み
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            self.get_logger().info(f"Model weights loaded from checkpoint (step {checkpoint.get('step', 'unknown')})")
        else:
            # 直接state_dictの場合
            model.load_state_dict(checkpoint)
            self.get_logger().info("Model weights loaded directly")
        
        return model
    
    def joint_state_callback(self, msg):
        """関節状態のコールバック"""
        # 関節位置の抽出
        name_to_pos = dict(zip(msg.name, msg.position))
        arm_pos = [name_to_pos.get(j, 0.0) for j in ARM_JOINT_NAMES]
        gripper_pos = name_to_pos.get(GRIPPER_JOINT_NAME, 0.0)
        current_state = np.array(arm_pos + [gripper_pos], dtype=np.float32)
        
        self.latest_joint_state = current_state
        
        # 履歴の更新
        self.joint_history.append(current_state)
        if len(self.joint_history) > self.max_history_length:
            self.joint_history.pop(0)
    
    def get_observation_tensor(self) -> Dict[str, torch.Tensor]:
        """観測テンソルの準備"""
        if self.latest_joint_state is None:
            return None
        
        # 履歴データの準備
        if len(self.joint_history) < self.max_history_length:
            # 履歴が不足している場合は最初の状態でパディング
            padding = [self.joint_history[0]] * (self.max_history_length - len(self.joint_history))
            history = padding + self.joint_history
        else:
            history = self.joint_history
        
        # テンソルに変換
        history_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 5)
        
        return {
            'observation.state': history_tensor,
            'observation.environment_state': history_tensor
        }
    
    def publish_action(self):
        """アクションの公開"""
        if self.latest_joint_state is None:
            return
        
        # 観測データの準備
        obs = self.get_observation_tensor()
        if obs is None:
            return
        
        # 推論実行
        with torch.no_grad():
            try:
                # モデルの順伝播
                output = self.policy(obs['observation.state'].to(self.device), return_uncertainty=True)
                action = output['action'].cpu().numpy().flatten()  # (5,)
                
                # 不確実性の取得（デバッグ用）
                uncertainty = output.get('uncertainty', None)
                if uncertainty is not None:
                    uncertainty = uncertainty.cpu().numpy().flatten()
                
                # フェーズ情報の取得（デバッグ用）
                phase_probs = output.get('phase_probs', None)
                if phase_probs is not None:
                    phase_probs = phase_probs.cpu().numpy()
                
                self.last_action = action
                
                # アクションの公開
                self.publish_arm_action(action[:4])
                self.publish_gripper_action(action[4])
                
                # デバッグ情報の出力
                self.log_action(action, uncertainty, phase_probs)
                
            except Exception as e:
                self.get_logger().error(f"Error during inference: {e}")
                # エラー時は前回のアクションを使用
                self.publish_arm_action(self.last_action[:4])
                self.publish_gripper_action(self.last_action[4])
    
    def publish_arm_action(self, arm_action: np.ndarray):
        """アームアクションの公開"""
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = arm_action.tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)
    
    def publish_gripper_action(self, gripper_action: float):
        """グリッパーアクションの公開"""
        jt_g = JointTrajectory()
        jt_g.joint_names = [GRIPPER_JOINT_NAME]
        pt_g = JointTrajectoryPoint()
        pt_g.positions = [float(gripper_action)]
        pt_g.time_from_start.sec = 0
        pt_g.time_from_start.nanosec = int(0.1 * 1e9)
        jt_g.points.append(pt_g)
        self.gripper_pub.publish(jt_g)
    
    def log_action(self, action: np.ndarray, uncertainty: Optional[np.ndarray] = None, phase_probs: Optional[np.ndarray] = None):
        """アクション情報のログ出力"""
        # 基本アクション情報
        log_msg = f"Action: arm[{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}] gripper[{action[4]:.3f}]"
        
        # 不確実性情報
        if uncertainty is not None:
            log_msg += f" | Uncertainty: {uncertainty.mean():.3f}"
        
        # フェーズ情報
        if phase_probs is not None:
            phase_names = ['pre_grasp', 'grasp', 'post_grasp', 'other']
            max_phase_idx = np.argmax(phase_probs[-1])  # 最新のタイムステップ
            max_phase_prob = phase_probs[-1][max_phase_idx]
            log_msg += f" | Phase: {phase_names[max_phase_idx]}({max_phase_prob:.2f})"
        
        print(log_msg)

def main():
    parser = argparse.ArgumentParser(description='CubeGraspPolicy inference node for CRANE+ V2 (ROS2)')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to trained CubeGraspPolicy model checkpoint (.pt file)')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to model configuration file (optional)')
    parser.add_argument('--history-length', type=int, default=32,
                       help='Length of joint state history for inference')
    
    args = parser.parse_args()
    
    # 引数の検証
    if not Path(args.model_path).exists():
        print(f"Error: Model path not found: {args.model_path}")
        return
    
    if args.config_path and not Path(args.config_path).exists():
        print(f"Warning: Config path not found: {args.config_path}, using default config")
        args.config_path = None
    
    # ROS2ノードの初期化
    rclpy.init()
    
    try:
        # 推論ノードの作成
        node = CubeGraspInferenceNode(args.model_path, args.config_path)
        
        # 履歴長の設定
        if hasattr(node, 'max_history_length'):
            node.max_history_length = args.history_length
        
        print(f"Starting CubeGraspPolicy inference...")
        print(f"Model: {args.model_path}")
        print(f"Config: {args.config_path}")
        print(f"History length: {args.history_length}")
        print("Press Ctrl+C to stop")
        
        # ノードの実行
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nInference stopped by user")
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        # クリーンアップ
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 