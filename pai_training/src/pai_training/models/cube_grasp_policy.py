#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class GraspAttention(nn.Module):
    """グリッパー状態に特化したアテンション機構"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # グリッパー状態の重み付け
        self.gripper_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, gripper_state: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # グリッパー状態を考慮した重み付け
        gripper_attention = torch.sigmoid(self.gripper_weight * gripper_state.unsqueeze(-1))
        x_weighted = x * gripper_attention
        
        # Multi-head attention
        q = self.q_proj(x_weighted).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_weighted).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_weighted).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim), dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(attention_output)

class GraspPhaseDetector(nn.Module):
    """グラスプフェーズ検出器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)  # pre-grasp, grasp, post-grasp, other
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out)

class CubeGraspPolicy(nn.Module):
    """Cube graspingタスクに特化したポリシーモデル"""
    
    def __init__(self, 
                 state_dim: int = 5,
                 action_dim: int = 5,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 use_grasp_attention: bool = True,
                 use_phase_detection: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_grasp_attention = use_grasp_attention
        self.use_phase_detection = use_phase_detection
        
        # 状態エンコーダー
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # グリッパー状態の特別な処理
        self.gripper_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)  # hidden_dimに合わせる
        )
        
        # 時系列処理用のLSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # グラスプアテンション
        if use_grasp_attention:
            self.grasp_attention = GraspAttention(hidden_dim)
        
        # フェーズ検出器
        if use_phase_detection:
            self.phase_detector = GraspPhaseDetector(state_dim)
            self.phase_encoder = nn.Linear(4, hidden_dim // 4)
        
        # アクション予測ヘッド
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 4 if use_phase_detection else 0), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # アクションの不確実性推定
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softplus()  # 正の値のみ
        )
        
        # 初期化
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, 
                state: torch.Tensor, 
                gripper_state: Optional[torch.Tensor] = None,
                return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: (batch_size, state_dim) or (batch_size, seq_len, state_dim)
            gripper_state: (batch_size, 1) or (batch_size, seq_len, 1) - グリッパー状態
            return_uncertainty: 不確実性を返すかどうか
        """
        # データの形状を確認・調整
        if len(state.shape) == 2:
            # 単一ステップの場合、シーケンスとして扱う
            state = state.unsqueeze(1)  # (batch_size, 1, state_dim)
            if gripper_state is not None:
                gripper_state = gripper_state.unsqueeze(1)  # (batch_size, 1, 1)
        
        batch_size, seq_len, _ = state.shape
        
        # グリッパー状態の抽出（最後の次元）
        if gripper_state is None:
            gripper_state = state[:, :, -1:].detach()
        
        # 状態エンコーディング
        state_features = self.state_encoder(state)
        
        # グリッパー状態の特別なエンコーディング
        gripper_features = self.gripper_encoder(gripper_state)
        
        # 状態とグリッパー特徴の結合
        combined_features = state_features + gripper_features
        
        # LSTM処理
        lstm_out, (h_n, c_n) = self.lstm(combined_features)
        
        # グラスプアテンション
        if self.use_grasp_attention:
            lstm_out = self.grasp_attention(lstm_out, gripper_state.squeeze(-1))
        
        # フェーズ検出
        phase_features = None
        if self.use_phase_detection:
            phase_logits = self.phase_detector(state)
            phase_probs = F.softmax(phase_logits, dim=-1)
            phase_features = self.phase_encoder(phase_probs)
        
        # 最終特徴量の結合
        if phase_features is not None:
            final_features = torch.cat([lstm_out, phase_features], dim=-1)
        else:
            final_features = lstm_out
        
        # アクション予測
        actions = self.action_head(final_features)
        
        # 出力の形状を調整（シーケンス長が1の場合は削除）
        if actions.shape[1] == 1:
            actions = actions.squeeze(1)  # (batch_size, action_dim)
        
        result = {"action": actions}
        
        # 不確実性推定
        if return_uncertainty:
            uncertainty = self.uncertainty_head(lstm_out)
            # 出力の形状を調整（シーケンス長が1の場合は削除）
            if uncertainty.shape[1] == 1:
                uncertainty = uncertainty.squeeze(1)  # (batch_size, action_dim)
            result["uncertainty"] = uncertainty
        
        # フェーズ情報も返す
        if self.use_phase_detection:
            result["phase_logits"] = phase_logits
            result["phase_probs"] = phase_probs
        
        return result

class CubeGraspPolicyWithCurriculum(nn.Module):
    """カリキュラム学習対応のCube graspingポリシー"""
    
    def __init__(self, 
                 state_dim: int = 5,
                 action_dim: int = 5,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 基本ポリシー
        self.base_policy = CubeGraspPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # カリキュラム学習用の補助タスク
        self.auxiliary_tasks = nn.ModuleDict({
            'grasp_prediction': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'trajectory_completion': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, state_dim)
            )
        })
        
        # カリキュラム重み
        self.curriculum_weights = nn.Parameter(torch.ones(len(self.auxiliary_tasks)))
        
    def forward(self, 
                state: torch.Tensor,
                gripper_state: Optional[torch.Tensor] = None,
                return_auxiliary: bool = False) -> Dict[str, torch.Tensor]:
        
        # 基本ポリシーの出力
        base_output = self.base_policy(state, gripper_state, return_uncertainty=True)
        
        # 補助タスクの出力
        if return_auxiliary:
            # LSTMの隠れ状態を取得
            lstm_out = self.base_policy.lstm(
                self.base_policy.state_encoder(state) + 
                self.base_policy.gripper_encoder(gripper_state if gripper_state is not None else state[:, :, -1:])
            )[0]
            
            auxiliary_outputs = {}
            for task_name, task_head in self.auxiliary_tasks.items():
                auxiliary_outputs[f'aux_{task_name}'] = task_head(lstm_out)
            
            base_output.update(auxiliary_outputs)
        
        return base_output

def create_cube_grasp_policy(config: Dict) -> nn.Module:
    """設定に基づいてCube graspingポリシーを作成"""
    
    policy_type = config.get('policy_type', 'standard')
    
    if policy_type == 'standard':
        return CubeGraspPolicy(
            state_dim=config.get('state_dim', 5),
            action_dim=config.get('action_dim', 5),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.1),
            use_grasp_attention=config.get('use_grasp_attention', True),
            use_phase_detection=config.get('use_phase_detection', True)
        )
    elif policy_type == 'curriculum':
        return CubeGraspPolicyWithCurriculum(
            state_dim=config.get('state_dim', 5),
            action_dim=config.get('action_dim', 5),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

# 使用例
if __name__ == "__main__":
    # 設定例
    config = {
        'policy_type': 'standard',
        'state_dim': 5,
        'action_dim': 5,
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'use_grasp_attention': True,
        'use_phase_detection': True
    }
    
    # モデル作成
    model = create_cube_grasp_policy(config)
    
    # テスト入力
    batch_size, seq_len = 4, 32
    state = torch.randn(batch_size, seq_len, 5)
    gripper_state = torch.randn(batch_size, seq_len, 1)
    
    # 順伝播
    output = model(state, gripper_state, return_uncertainty=True)
    
    print(f"Action shape: {output['action'].shape}")
    print(f"Uncertainty shape: {output['uncertainty'].shape}")
    if 'phase_probs' in output:
        print(f"Phase probs shape: {output['phase_probs'].shape}") 