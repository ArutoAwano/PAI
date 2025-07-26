features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
}
# Below is used for the dataset conversion
robot_type = "crane_plus"
repo_id = "crane_plus_pekori"
root = "/home/ubuntu/dataset/lerobot_dataset/crane_plus_pekori"
task_name = "pekori"
