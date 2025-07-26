# Physical AI Imitation Learning excercise

## Features
- rosbag2lerobot: Dataset conversion from rosbag2 to LeRobotDataset v2.1 format
- training: Imitation Learning using LeRobot

## Installation
Clone the repo
```bash
cd && git clone --recursive https://github.com/matsuolab/pai_training
cd pai_training
```
Get [pixi](https://pixi.sh/latest/) and install all the dependencies
```bash
curl -fsSL https://pixi.sh/install.sh | bash
echo 'eval "$(pixi completion --shell bash)"' >> ~/.bashrc
source ~/.bashrc
```
```bash
pixi install
```

## Usage
Convert rosbag2 to LeRobotDataset v2.1 format
```bash
pixi run convert
```
Visualize the dataset
```bash
pixi run python scripts/visualize_dataset.py --root /home/ubuntu/dataset/lerobot_dataset/crane_plus_pekori --repo-id crane_plus_pekori --episode-index 1 --mode distant
# Copy the output from above command and paste it in your *local* web browser (it may take a second to load)
```
Train the policy
```bash
pixi run wandb login # login to wandb, access https://wandb.ai/authorize to get your API key
# Invoke training with the config in [configs/train_config.yaml](configs/train_config.yaml)
pixi run train
```
Run inference.  
It takes the model path as an argument.
```bash
pixi run infer --model-path /home/ubuntu/checkpoints/train/crane_plus_pekori_act
```
Debug policy.  
This script evaluates the policy on the specified dataset offline.  
This is useful to check if the policy is working as expected.  
You can separate the dataset into training and evaluation sets to evaluate the policy in more detail.
```bash
pixi run debug-policy --dataset-root /home/ubuntu/dataset/lerobot_dataset/crane_plus_pekori --repo-id crane_plus_pekori --policy-path /home/ubuntu/checkpoints/train/crane_plus_pekori_act --episode-index 1
```

## Pixi help
Visit [pixi](https://pixi.sh/latest/) for more details.

### Environment Management
```bash
# Cleanup environment
pixi clean

# Add dependencies
pixi list  # show all installed packages

# Package Management
# PyPI (pip) packages
pixi add --pypi package  # install
pixi remove --pypi package  # uninstall

# conda packages
pixi add package  # install
pixi remove package  # uninstall

# Activate environment
pixi shell  # enter pixi's virtual environment shell (like source env/bin/activate in venv)
# or
pixi run "..."  # run command in default shell
```

### Task Management
Tasks can be defined in `pyproject.toml` under `[tool.pixi.tasks]`:
```toml
[tool.pixi.tasks]
my-task = "python my_script.py"  # simple command
```

You can then run these tasks using:
```bash
pixi run my-task  #= pixi shell & python my_script.py
pixi run python my_script.py  # run a command in the pixi environment
```
