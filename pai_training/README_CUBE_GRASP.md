# Cube Grasping Task - Enhanced Training Pipeline

このプロジェクトは、cube grasping タスクのための改良されたデータ拡張とモデルアーキテクチャを提供します。

## 🚀 主な改良点

### 1. データ拡張 (Data Augmentation)

- **時間歪曲 (Time Warping)**: 軌道の時間軸をランダムに歪曲
- **関節制限を考慮したノイズ**: ロボットの関節制限を考慮した安全なノイズ追加
- **アクション平滑化**: アクション軌道の平滑化による自然な動作生成
- **グラスプフェーズ拡張**: グリッパーの開閉タイミングに基づく特別な拡張
- **軌道補間**: より細かい時間ステップでの補間

### 2. モデルアーキテクチャの改善

- **グラスプアテンション**: グリッパー状態に特化したアテンション機構
- **フェーズ検出器**: グラスプの各フェーズ（pre-grasp, grasp, post-grasp）を自動検出
- **不確実性推定**: ベイズ的アプローチによるアクションの不確実性推定
- **カリキュラム学習**: 段階的な学習による効率的な訓練

## 📁 ファイル構成

```
pai_training/
├── scripts/
│   ├── rosbag2lerobot_with_augmentation.py  # データ拡張付き変換スクリプト
│   ├── train_cube_grasp.py                  # 改良された学習スクリプト
│   └── infer_cube_grasp.py                  # 改良された推論スクリプト
├── src/pai_training/models/
│   └── cube_grasp_policy.py                 # 改良されたポリシーモデル
├── configs/
│   ├── cube_grasp_config.yaml               # 新しい設定ファイル
│   ├── crane_features.py                    # 既存の特徴量設定
│   └── conversion_config.yaml               # 既存の変換設定
└── README_CUBE_GRASP.md                     # このファイル
```

## 🛠️ 使用方法

### 1. データ拡張付きの変換

```bash
# pixiを使用してデータ拡張付きで変換
pixi run convert-augmented

# または直接実行
python scripts/rosbag2lerobot_with_augmentation.py
```

**設定例** (`configs/conversion_config.yaml`):

```yaml
bag_dir: "/home/ubuntu/dataset/rosbags/pekori"
output_dir: "/home/ubuntu/dataset/lerobot_dataset/cube_grasp_enhanced"
config: "configs/crane_features.py"
target_freq: 20
```

### 2. 改良されたモデルの学習

````bash
# pixiを使用して学習
pixi run train-cube-grasp

# カスタム設定ファイル
pixi run train-cube-grasp -- --config configs/cube_grasp_config.yaml

# カスタム出力ディレクトリ
pixi run train-cube-grasp -- --config configs/cube_grasp_config.yaml --output_dir /home/ubuntu/checkpoints/train/cube_grasp_enhanced

# GPU指定
pixi run train-cube-grasp -- --config configs/cube_grasp_config.yaml --device cuda

# または直接実行
python scripts/train_cube_grasp.py --config configs/cube_grasp_config.yaml

### 3. 改良されたモデルの推論

```bash
# pixiを使用して推論
pixi run infer-cube-grasp -- --model-path /home/ubuntu/checkpoints/train/cube_grasp_enhanced/latest_checkpoint.pt

# カスタム設定ファイル付き
pixi run infer-cube-grasp -- --model-path /home/ubuntu/checkpoints/train/cube_grasp_enhanced/latest_checkpoint.pt --config-path configs/cube_grasp_config.yaml

# 履歴長の調整
pixi run infer-cube-grasp -- --model-path /home/ubuntu/checkpoints/train/cube_grasp_enhanced/latest_checkpoint.pt --history-length 64

# または直接実行
python scripts/infer_cube_grasp.py --model-path /home/ubuntu/checkpoints/train/cube_grasp_enhanced/latest_checkpoint.pt
````

## ⚙️ 設定パラメータ

### データ拡張設定

```yaml
augmentation_config:
  noise_std: 0.01 # ガウシアンノイズの標準偏差
  time_warp_factor: 0.1 # 時間歪曲の強度
  action_noise_std: 0.005 # アクションへのノイズ
  joint_limit_buffer: 0.05 # 関節制限の安全マージン
  enable_augmentation: true # 拡張の有効/無効
  augmentation_factor: 3 # データ拡張倍率
```

### モデル設定

```yaml
model_config:
  policy_type: "standard" # "standard" または "curriculum"
  state_dim: 5 # 状態次元
  action_dim: 5 # アクション次元
  hidden_dim: 256 # 隠れ層の次元
  num_layers: 4 # LSTM層数
  dropout: 0.1 # Dropout率
  use_grasp_attention: true # グラスプアテンションの使用
  use_phase_detection: true # フェーズ検出の使用
```

### カリキュラム学習設定

```yaml
curriculum_config:
  curriculum_steps:
    - [100, 0.1] # 100ステップ後、補助タスク重み0.1
    - [500, 0.5] # 500ステップ後、補助タスク重み0.5
    - [1000, 1.0] # 1000ステップ後、補助タスク重み1.0
```

## 🔧 カスタマイズ

### 新しいデータ拡張手法の追加

`scripts/rosbag2lerobot_with_augmentation.py`の`DataAugmentation`クラスに新しいメソッドを追加：

```python
def your_new_augmentation(self, data: np.ndarray) -> np.ndarray:
    """新しい拡張手法"""
    # 実装
    return augmented_data
```

### 新しいモデルアーキテクチャの追加

`src/pai_training/models/cube_grasp_policy.py`に新しいクラスを追加し、`create_cube_grasp_policy`関数を更新。

## 📊 監視とログ

### Weights & Biases 統合

設定で`use_wandb: true`にすると、以下のメトリクスが自動的に記録されます：

- **損失**: `action_loss`, `phase_loss`, `uncertainty_reg`
- **検証メトリクス**: `val_action_loss`, `val_phase_loss`
- **学習率**: `learning_rate`
- **カリキュラム重み**: `curriculum_weights`

### チェックポイント

- 定期的にチェックポイントが保存されます
- `latest_checkpoint.pt`で最新の状態を保存
- 学習再開時に自動的に最新チェックポイントから開始

## 🎯 期待される改善効果

1. **データ効率の向上**: データ拡張により少ないデータでも高精度な学習
2. **ロバスト性の向上**: ノイズや時間変化に対する耐性
3. **グラスプ精度の向上**: フェーズ検出による適切なタイミングでのグラスプ
4. **学習の安定性**: カリキュラム学習による段階的な学習
5. **不確実性の定量化**: ベイズ的アプローチによる信頼性の評価

## 🐛 トラブルシューティング

### よくある問題

1. **メモリ不足**

   - `batch_size`を小さくする
   - `num_workers`を減らす

2. **学習が収束しない**

   - `learning_rate`を調整
   - `augmentation_factor`を減らす

3. **データ拡張が強すぎる**
   - `noise_std`や`time_warp_factor`を小さくする

### デバッグモード

```bash
# デバッグ用の設定で実行
python scripts/train_cube_grasp.py --config configs/cube_grasp_config.yaml --debug
```

## 📝 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 🤝 貢献

バグ報告や機能要求は、GitHub の Issues ページでお知らせください。
