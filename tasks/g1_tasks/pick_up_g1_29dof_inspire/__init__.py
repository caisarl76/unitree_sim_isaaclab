# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
import gymnasium as gym

from . import pickup_g1_29dof_inspire_joint_env_cfg

gym.register(
    id="Isaac-PickUp-RedBlock-G129-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickup_g1_29dof_inspire_joint_env_cfg.PickUpG129InspireEnvCfg,
    },
    disable_env_checker=True,
)
