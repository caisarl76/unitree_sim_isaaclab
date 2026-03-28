import gymnasium as gym
from . import pickplace_toastedbread_g1_29dof_dex3_joint_env_cfg

gym.register(
    id="Isaac-PickPlace-ToastedBread-G129-Dex3-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_toastedbread_g1_29dof_dex3_joint_env_cfg.PickPlaceG129DEX3ToastedBreadEnvCfg,
    },
    disable_env_checker=True,
)
