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
        self._debug_output = False  # デバッグ出力フラグ
        self._phase_shape_logged = False  # フェーズ形状ログフラグ
        
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
        try:
            # 関節位置の抽出
            name_to_pos = dict(zip(msg.name, msg.position))
            arm_pos = [float(name_to_pos.get(j, 0.0)) for j in ARM_JOINT_NAMES]
            gripper_pos = float(name_to_pos.get(GRIPPER_JOINT_NAME, 0.0))
            current_state = np.array(arm_pos + [gripper_pos], dtype=np.float32)
            
            self.latest_joint_state = current_state
            
            # 履歴に追加
            self.joint_history.append(current_state.copy())
            
            # 履歴の長さを制限
            if len(self.joint_history) > self.max_history_length:
                self.joint_history = self.joint_history[-self.max_history_length:]
                
        except Exception as e:
            self.get_logger().error(f"Error in joint_state_callback: {e}")
    
    def get_observation_tensor(self) -> Dict[str, torch.Tensor]:
        """観測テンソルの準備"""
        if self.latest_joint_state is None:
            return None
        
        # 履歴データの準備
        if len(self.joint_history) < self.max_history_length:
            # 履歴が不足している場合は現在の状態でパディング
            if len(self.joint_history) > 0:
                padding = [self.joint_history[0]] * (self.max_history_length - len(self.joint_history))
                history = padding + self.joint_history
            else:
                # 履歴が空の場合は現在の状態で埋める
                history = [self.latest_joint_state.astype(np.float32)] * self.max_history_length
        else:
            history = self.joint_history[-self.max_history_length:]  # 最新の履歴のみ使用
        
        # numpy配列をfloat32に変換してからテンソルに変換
        history_float32 = [h.astype(np.float32) for h in history]
        # リストを単一のnumpy配列に変換してからテンソルに変換（警告を回避）
        history_array = np.array(history_float32, dtype=np.float32)
        history_tensor = torch.from_numpy(history_array).unsqueeze(0)  # (1, seq_len, 5)
        
        return {
            'observation.state': history_tensor,
            'observation.environment_state': history_tensor
        }
    
    def publish_action(self):
        """アクションの公開"""
        if self.latest_joint_state is None:
            return
        
        # 元のinfer.pyと同じ観測データの準備
        obs = {
            'observation.state': torch.tensor(self.latest_joint_state, dtype=torch.float32).unsqueeze(0),  # (1, 5)
            'observation.environment_state': torch.tensor(self.latest_joint_state, dtype=torch.float32).unsqueeze(0),  # (1, 5)
        }
        
        # 推論実行
        with torch.no_grad():
            try:
                # モデルの順伝播（元のinfer.pyと同じ方法）
                output = self.policy(obs['observation.state'].to(self.device), return_uncertainty=True)
                action = output['action'].cpu().numpy().flatten().astype(np.float32)  # (5,)
                
                # デバッグ用：出力構造を確認
                if hasattr(self, '_debug_output') and not self._debug_output:
                    self.get_logger().info(f"Model output keys: {list(output.keys())}")
                    self._debug_output = True
                
                # 不確実性とフェーズ情報の取得（オプション）
                uncertainty = None
                phase_probs = None
                try:
                    # 不確実性の取得
                    uncertainty = output.get('uncertainty', None)
                    if uncertainty is not None:
                        uncertainty = uncertainty.cpu().numpy().flatten().astype(np.float32)
                    
                    # フェーズ情報の取得
                    phase_probs = output.get('phase_probs', None)
                    if phase_probs is not None:
                        phase_probs = phase_probs.cpu().numpy().astype(np.float32)
                        # 推論時は最新のタイムステップのみを使用
                        if phase_probs.ndim == 3:  # (batch_size, seq_len, num_phases)
                            phase_probs = phase_probs[0, -1, :]  # (num_phases,)
                        elif phase_probs.ndim == 2:  # (seq_len, num_phases)
                            phase_probs = phase_probs[-1, :]  # (num_phases,)
                        # デバッグ用：フェーズ確率の形状をログ出力
                        if not hasattr(self, '_phase_shape_logged'):
                            self.get_logger().info(f"Phase probs shape: {phase_probs.shape}, values: {phase_probs}")
                            self._phase_shape_logged = True
                except Exception as e:
                    self.get_logger().warn(f"Could not get uncertainty/phase info: {e}")
                
                self.last_action = action
                
                # アクションの公開（元のinfer.pyと同じ方法）
                jt = JointTrajectory()
                jt.joint_names = ARM_JOINT_NAMES
                pt = JointTrajectoryPoint()
                pt.positions = action[:4].tolist()
                pt.time_from_start.sec = 0
                pt.time_from_start.nanosec = int(0.1 * 1e9)
                jt.points.append(pt)
                self.arm_pub.publish(jt)
                
                jt_g = JointTrajectory()
                jt_g.joint_names = [GRIPPER_JOINT_NAME]
                pt_g = JointTrajectoryPoint()
                pt_g.positions = [float(action[4])]
                pt_g.time_from_start.sec = 0
                pt_g.time_from_start.nanosec = int(0.1 * 1e9)
                jt_g.points.append(pt_g)
                self.gripper_pub.publish(jt_g)
                
                # デバッグ情報の出力
                self.log_action(action, uncertainty, phase_probs)
                
            except Exception as e:
                import traceback
                self.get_logger().error(f"Error during inference: {e}")
                self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                # エラー時は前回のアクションを使用
                if self.last_action is not None:
                    # 元のinfer.pyと同じ方法でアクションを公開
                    jt = JointTrajectory()
                    jt.joint_names = ARM_JOINT_NAMES
                    pt = JointTrajectoryPoint()
                    pt.positions = self.last_action[:4].tolist()
                    pt.time_from_start.sec = 0
                    pt.time_from_start.nanosec = int(0.1 * 1e9)
                    jt.points.append(pt)
                    self.arm_pub.publish(jt)
                    
                    jt_g = JointTrajectory()
                    jt_g.joint_names = [GRIPPER_JOINT_NAME]
                    pt_g = JointTrajectoryPoint()
                    pt_g.positions = [float(self.last_action[4])]
                    pt_g.time_from_start.sec = 0
                    pt_g.time_from_start.nanosec = int(0.1 * 1e9)
                    jt_g.points.append(pt_g)
                    self.gripper_pub.publish(jt_g)
    
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
        # numpy配列をfloatに変換して安全にフォーマット
        action_float = action.astype(np.float32)
        
        # 基本アクション情報
        log_msg = f"Action: arm[{action_float[0]:.3f}, {action_float[1]:.3f}, {action_float[2]:.3f}, {action_float[3]:.3f}] gripper[{action_float[4]:.3f}]"
        
        # 不確実性情報
        if uncertainty is not None:
            uncertainty_float = uncertainty.astype(np.float32)
            log_msg += f" | Uncertainty: {float(uncertainty_float.mean()):.3f}"
        
        # フェーズ情報
        if phase_probs is not None:
            try:
                phase_names = ['pre_grasp', 'grasp', 'post_grasp', 'other']
                phase_probs_float = phase_probs.astype(np.float32)
                
                # フェーズ確率の形状を確認して適切に処理
                if phase_probs_float.ndim == 0:
                    # スカラー値の場合
                    max_phase_prob = float(phase_probs_float)
                    max_phase_idx = 0
                elif phase_probs_float.ndim == 1:
                    # 1次元の場合（単一のフェーズ確率）
                    if len(phase_probs_float) == 1:
                        # 長さ1の配列
                        max_phase_prob = float(phase_probs_float[0])
                        max_phase_idx = 0
                    else:
                        # 複数のフェーズ確率
                        # 安全なargmax処理
                        max_val = float(phase_probs_float.max())
                        max_indices = np.where(phase_probs_float == max_val)[0]
                        max_phase_idx = int(max_indices[0])
                        max_phase_prob = max_val
                elif phase_probs_float.ndim == 2:
                    # 2次元の場合（時系列のフェーズ確率）
                    if phase_probs_float.shape[0] == 1:
                        # 単一のタイムステップ
                        max_val = float(phase_probs_float[0].max())
                        max_indices = np.where(phase_probs_float[0] == max_val)[0]
                        max_phase_idx = int(max_indices[0])
                        max_phase_prob = max_val
                    else:
                        # 複数のタイムステップ
                        max_val = float(phase_probs_float[-1].max())
                        max_indices = np.where(phase_probs_float[-1] == max_val)[0]
                        max_phase_idx = int(max_indices[0])
                        max_phase_prob = max_val
                else:
                    # その他の形状の場合は平均を取る
                    mean_probs = phase_probs_float.mean(axis=0) if phase_probs_float.ndim > 1 else phase_probs_float
                    max_val = float(mean_probs.max())
                    max_indices = np.where(mean_probs == max_val)[0]
                    max_phase_idx = int(max_indices[0])
                    max_phase_prob = max_val
                
                # インデックスが範囲内かチェック
                if max_phase_idx < len(phase_names):
                    log_msg += f" | Phase: {phase_names[max_phase_idx]}({max_phase_prob:.2f})"
                else:
                    log_msg += f" | Phase: unknown({max_phase_prob:.2f})"
            except Exception as e:
                log_msg += f" | Phase: error({str(e)[:20]})"
        
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
    
    node = None
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
        try:
            if node is not None:
                node.destroy_node()
        except Exception as e:
            print(f"Error destroying node: {e}")
        
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {e}")

if __name__ == '__main__':
    main() 