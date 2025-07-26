#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
try:
    import wandb
except ImportError:
    wandb = None
    print("Warning: wandb not available. Logging will be disabled.")
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Progress bars will be disabled.")
    tqdm = lambda x: x
import random

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pai_training.models.cube_grasp_policy import create_cube_grasp_policy

class CubeGraspTrainer:
    """Cube graspingタスク用のトレーナー"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # データセットの読み込み
        self.dataset = LeRobotDataset(
            repo_id=config['repo_id'],
            root=config['dataset_root'],
            split='train'
        )
        
        # モデルの作成
        self.model = create_cube_grasp_policy(config['model_config'])
        self.model.to(self.device)
        
        # オプティマイザーとスケジューラー
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('training_steps', 1000),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # 損失関数
        self.action_loss_fn = nn.MSELoss()
        self.phase_loss_fn = nn.CrossEntropyLoss()
        
        # カリキュラム学習の設定
        self.curriculum_config = config.get('curriculum_config', {})
        self.current_curriculum_step = 0
        
        # ログ設定
        self.log_freq = config.get('log_freq', 10)
        self.save_freq = config.get('save_freq', 100)
        
        # 出力ディレクトリ
        self.output_dir = Path(config['output_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # wandb初期化
        if config.get('use_wandb', False) and wandb is not None:
            wandb.init(
                project=config.get('wandb_project', 'cube-grasp'),
                config=config,
                name=config.get('experiment_name', 'cube-grasp-experiment')
            )
        elif config.get('use_wandb', False) and wandb is None:
            print("Warning: wandb is enabled in config but not available. Disabling wandb logging.")
            config['use_wandb'] = False
    
    def create_dataloader(self) -> DataLoader:
        """データローダーの作成"""
        return DataLoader(
            self.dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
    
    def compute_loss(self, 
                    pred_actions: torch.Tensor,
                    target_actions: torch.Tensor,
                    pred_phases: Optional[torch.Tensor] = None,
                    target_phases: Optional[torch.Tensor] = None,
                    pred_uncertainty: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """損失の計算"""
        losses = {}
        
        # アクション損失
        action_loss = self.action_loss_fn(pred_actions, target_actions)
        losses['action_loss'] = action_loss
        
        # 不確実性を考慮した損失（ベイズ的アプローチ）
        if pred_uncertainty is not None:
            # 不確実性の逆数を重みとして使用
            uncertainty_weights = 1.0 / (pred_uncertainty + 1e-6)
            weighted_action_loss = torch.mean(uncertainty_weights * (pred_actions - target_actions) ** 2)
            losses['weighted_action_loss'] = weighted_action_loss
            
            # 不確実性の正則化
            uncertainty_reg = torch.mean(pred_uncertainty)
            losses['uncertainty_reg'] = uncertainty_reg
        
        # フェーズ分類損失
        if pred_phases is not None and target_phases is not None:
            phase_loss = self.phase_loss_fn(pred_phases.view(-1, pred_phases.size(-1)), 
                                          target_phases.view(-1))
            losses['phase_loss'] = phase_loss
        
        # カリキュラム学習の補助タスク損失
        if hasattr(self.model, 'auxiliary_tasks'):
            aux_losses = {}
            for task_name, task_head in self.model.auxiliary_tasks.items():
                if f'aux_{task_name}' in batch:
                    aux_pred = batch[f'aux_{task_name}']
                    aux_target = batch[f'target_{task_name}']
                    
                    if task_name == 'grasp_prediction':
                        aux_loss = nn.BCELoss()(aux_pred, aux_target)
                    else:
                        aux_loss = nn.MSELoss()(aux_pred, aux_target)
                    
                    aux_losses[f'aux_{task_name}_loss'] = aux_loss
            
            # カリキュラム重みを適用
            for loss_name, loss_value in aux_losses.items():
                weight = self.model.curriculum_weights[list(aux_losses.keys()).index(loss_name)]
                losses[loss_name] = weight * loss_value
        
        return losses
    
    def update_curriculum(self, step: int):
        """カリキュラム学習の更新"""
        if not self.curriculum_config:
            return
        
        # 段階的に補助タスクの重みを調整
        curriculum_steps = self.curriculum_config.get('curriculum_steps', [])
        for i, (step_threshold, weight) in enumerate(curriculum_steps):
            if step >= step_threshold and hasattr(self.model, 'curriculum_weights'):
                self.model.curriculum_weights.data[i] = weight
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """1ステップの学習"""
        self.model.train()
        
        # データをデバイスに移動
        state = batch['observation.state'].to(self.device)
        action = batch['action'].to(self.device)
        
        # グリッパー状態の抽出
        gripper_state = state[:, :, -1:].detach()
        
        # 順伝播
        output = self.model(state, gripper_state, return_uncertainty=True)
        
        # 損失計算
        losses = self.compute_loss(
            pred_actions=output['action'],
            target_actions=action,
            pred_phases=output.get('phase_logits'),
            pred_uncertainty=output.get('uncertainty')
        )
        
        # 総損失
        total_loss = sum(losses.values())
        
        # 逆伝播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                     self.config.get('max_grad_norm', 1.0))
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                state = batch['observation.state'].to(self.device)
                action = batch['action'].to(self.device)
                gripper_state = state[:, :, -1:].detach()
                
                output = self.model(state, gripper_state, return_uncertainty=True)
                
                losses = self.compute_loss(
                    pred_actions=output['action'],
                    target_actions=action,
                    pred_phases=output.get('phase_logits'),
                    pred_uncertainty=output.get('uncertainty')
                )
                
                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item()
                
                num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """チェックポイントの保存"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_step_{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 最新のチェックポイントも保存
        latest_path = self.output_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
    
    def train(self):
        """学習の実行"""
        dataloader = self.create_dataloader()
        
        # 検証用データローダー
        val_dataset = LeRobotDataset(
            repo_id=self.config['repo_id'],
            root=self.config['dataset_root'],
            split='val'
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"Starting training for {self.config.get('training_steps', 1000)} steps")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for step in tqdm(range(self.config.get('training_steps', 1000))):
            # カリキュラム学習の更新
            self.update_curriculum(step)
            
            # 学習ステップ
            batch = next(iter(dataloader))
            train_losses = self.train_step(batch)
            
            # ログ出力
            if step % self.log_freq == 0:
                # 検証
                val_losses = self.validate(val_dataloader)
                
                # メトリクスの記録
                metrics = {**train_losses, **{f'val_{k}': v for k, v in val_losses.items()}}
                metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
                metrics['step'] = step
                
                print(f"Step {step}: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
                
                if self.config.get('use_wandb', False) and wandb is not None:
                    wandb.log(metrics)
            
            # チェックポイント保存
            if step % self.save_freq == 0:
                self.save_checkpoint(step, metrics)
        
        print("Training completed!")
        self.save_checkpoint(self.config.get('training_steps', 1000), metrics)

def main():
    parser = argparse.ArgumentParser(description='Train Cube Grasp Policy')
    parser.add_argument('--config', type=str, default='configs/cube_grasp_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    
    args = parser.parse_args()
    
    # 設定ファイルの存在確認
    if not Path(args.config).exists():
        print(f"Config file {args.config} not found. Please create it first.")
        print("You can copy from configs/cube_grasp_config.yaml.example if available.")
        return
    
    # 設定の読み込み
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # コマンドライン引数で上書き
    if args.output_dir:
        config['output_directory'] = args.output_dir
    if args.device:
        config['device'] = args.device
    
    # モデル設定の追加
    config['model_config'] = {
        'policy_type': 'standard',
        'state_dim': 5,
        'action_dim': 5,
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'use_grasp_attention': True,
        'use_phase_detection': True
    }
    
    # カリキュラム学習設定
    config['curriculum_config'] = {
        'curriculum_steps': [
            (100, 0.1),   # 100ステップ後、補助タスク重み0.1
            (500, 0.5),   # 500ステップ後、補助タスク重み0.5
            (1000, 1.0),  # 1000ステップ後、補助タスク重み1.0
        ]
    }
    
    # トレーナーの作成と学習実行
    trainer = CubeGraspTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 