import argparse
import torch
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
import matplotlib.pyplot as plt

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--repo-id', type=str, required=True, help='Dataset repo_id')
    parser.add_argument('--policy-path', type=str, required=True, help='Path to trained policy directory')
    parser.add_argument('--episode-index', type=int, default=0, help='Episode index to evaluate')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Load the dataset
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    print(f"Loaded dataset: {len(dataset)} samples, {dataset.num_episodes} episodes")

    # Sample one episode
    sampler = EpisodeSampler(dataset, args.episode_index)

    # Load the policy
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    # Evaluate the policy
    errors = []
    pred_actions = []
    gt_actions = []
    for i in sampler:
        sample = dataset[i]
        obs = {
            "observation.state": torch.tensor(sample["observation.state"], dtype=torch.float32).unsqueeze(0).to(args.device),
            "observation.environment_state": torch.tensor(sample["observation.environment_state"], dtype=torch.float32).unsqueeze(0).to(args.device),
        }
        gt_action = torch.tensor(sample["action"], dtype=torch.float32).to(args.device)

        with torch.no_grad():
            pred_action = policy.select_action(obs).squeeze(0)

        print(f"\033[32m pred_action: {pred_action}\033[0m")
        print(f"\033[31m gt_action: {gt_action}\033[0m")

        error = torch.norm(pred_action - gt_action).item()
        errors.append(error)
        pred_actions.append(pred_action.cpu().numpy())
        gt_actions.append(gt_action.cpu().numpy())

    print(f"\nMean L2 error for episode {args.episode_index} over {len(errors)} frames: {sum(errors)/len(errors):.4f}")

    # Plotting
    pred_actions = torch.tensor(pred_actions)
    gt_actions = torch.tensor(gt_actions)
    num_joints = pred_actions.shape[1] if pred_actions.ndim > 1 else 1
    frames = range(len(pred_actions))
    plt.figure(figsize=(10, 6))
    for j in range(num_joints):
        plt.plot(frames, gt_actions[:, j], label=f"GT action {j}", linestyle='--')
        plt.plot(frames, pred_actions[:, j], label=f"Pred action {j}")
    plt.xlabel("Frame")
    plt.ylabel("Action Value")
    plt.title(f"GT vs Predicted Actions for Episode {args.episode_index}")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
