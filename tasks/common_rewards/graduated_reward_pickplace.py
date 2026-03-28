# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Graduated reward for pick-up tasks.

4-tier reward providing continuous signal for manipulation progress:
  Tier 1 (0.25): Hand reaches near object (within reach_threshold)
  Tier 2 (0.50): Object grasped — lifted off table surface
  Tier 3 (0.75): Object lifted significantly above table
  Tier 4 (1.00): Object in relaxed target zone

Replaces base_reward_pickplace_redblock.py which uses a tiny post target
(3.7cm x 4.85cm, height +/-2.5mm) that gave 0% for all models.
"""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _find_wrist_body_index(body_names: list[str], preferred: str, side: str) -> int:
    """Find wrist body index, with fallback search if preferred name not found."""
    if preferred in body_names:
        return body_names.index(preferred)
    # Fallback: search for any body containing side + "wrist"
    for i, name in enumerate(body_names):
        if side in name and "wrist" in name and "yaw" in name:
            return i
    for i, name in enumerate(body_names):
        if side in name and "wrist" in name:
            return i
    raise ValueError(
        f"Could not find {side} wrist body. Preferred: '{preferred}'. "
        f"Available: {body_names}"
    )


def compute_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    # Tier 1: reach
    reach_threshold: float = 0.10,          # 10cm hand-to-object distance
    # Tier 2: grasp (object lifted off table)
    grasp_height_above_table: float = 0.02, # 2cm above table surface
    # Tier 3: lift
    lift_height_above_table: float = 0.05,  # 5cm above table surface
    # Tier 4: place in target zone (relaxed 15cm x 15cm)
    target_min_x: float = -4.35,
    target_max_x: float = -4.05,
    target_min_y: float = -4.13,
    target_max_y: float = -3.83,
    target_min_height_above_table: float = 0.05,
    # Table surface height (from scene: object at 0.84, cube half-height 0.03)
    table_height: float = 0.81,
    # Wrist body names for reach detection (with fallback search)
    left_wrist_body: str = "left_wrist_yaw_link",
    right_wrist_body: str = "right_wrist_yaw_link",
) -> torch.Tensor:
    """Compute graduated pick-up reward.

    Args:
        env: The Isaac Lab environment.
        object_cfg: Scene entity config for the manipulated object.
        Other args: Threshold parameters for each tier.

    Returns:
        Tensor of shape (num_envs,) with reward values in {0.0, 0.25, 0.50, 0.75, 1.0}.
    """
    # Get object position [num_envs, 3]
    object_entity = env.scene[object_cfg.name]
    object_pos = object_entity.data.root_pos_w[:, :3]

    # Get robot wrist positions
    robot = env.scene["robot"]
    body_names = robot.data.body_names
    left_idx = _find_wrist_body_index(body_names, left_wrist_body, "left")
    right_idx = _find_wrist_body_index(body_names, right_wrist_body, "right")
    left_wrist_pos = robot.data.body_pos_w[:, left_idx, :3]
    right_wrist_pos = robot.data.body_pos_w[:, right_idx, :3]

    # Use closest wrist to object for reach detection
    left_dist = torch.norm(left_wrist_pos - object_pos, dim=-1)
    right_dist = torch.norm(right_wrist_pos - object_pos, dim=-1)
    hand_to_object = torch.min(left_dist, right_dist)

    # Object height above table
    object_height_above_table = object_pos[:, 2] - table_height

    # Initialize reward to 0
    num_envs = object_pos.shape[0]
    reward = torch.zeros(num_envs, device=object_pos.device)

    # Tier 1 (0.25): hand reaches near object
    reached = hand_to_object < reach_threshold
    reward = torch.where(reached, torch.tensor(0.25, device=reward.device), reward)

    # Tier 2 (0.50): object lifted off table (grasped)
    grasped = object_height_above_table > grasp_height_above_table
    reward = torch.where(grasped, torch.tensor(0.50, device=reward.device), reward)

    # Tier 3 (0.75): object lifted significantly
    lifted = object_height_above_table > lift_height_above_table
    reward = torch.where(lifted, torch.tensor(0.75, device=reward.device), reward)

    # Tier 4 (1.00): object in relaxed target zone
    in_target = (
        (object_pos[:, 0] > target_min_x)
        & (object_pos[:, 0] < target_max_x)
        & (object_pos[:, 1] > target_min_y)
        & (object_pos[:, 1] < target_max_y)
        & (object_height_above_table > target_min_height_above_table)
    )
    reward = torch.where(in_target, torch.tensor(1.0, device=reward.device), reward)

    return reward
