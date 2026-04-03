# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""5-tier graduated reward for pick-AND-place tasks.

Extends the 4-tier pick-up reward with place detection:
  Tier 1 (0.2): Hand reaches near object
  Tier 2 (0.4): Object grasped — lifted off table
  Tier 3 (0.6): Object lifted significantly above table
  Tier 4 (0.8): Object within proximity of target area (XY)
  Tier 5 (1.0): Object placed on target — low height + hand released
"""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _find_wrist_body_index(body_names: list[str], preferred: str, side: str) -> int:
    """Find wrist body index, with fallback search if preferred name not found."""
    if preferred in body_names:
        return body_names.index(preferred)
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
    reach_threshold: float = 0.07,
    # Tier 2: grasp
    grasp_height_above_table: float = 0.06,
    # Tier 3: lift
    lift_height_above_table: float = 0.10,
    # Tier 4: near target (XY distance)
    target_xy_threshold: float = 0.05,
    # Tier 5: placed (object on target + hand released)
    place_height_above_table: float = 0.08,
    release_distance: float = 0.10,
    # Target position (XY) — matches red tape in scene config
    target_x: float = -4.21,
    target_y: float = -3.664,
    # Table surface height
    table_height: float = 0.81,
    # Wrist body names
    left_wrist_body: str = "left_wrist_yaw_link",
    right_wrist_body: str = "right_wrist_yaw_link",
) -> torch.Tensor:
    """Compute 5-tier graduated pick-and-place reward.

    Returns:
        Tensor of shape (num_envs,) with values in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
    """
    object_entity = env.scene[object_cfg.name]
    object_pos = object_entity.data.root_pos_w[:, :3]

    target_pos = torch.tensor(
        [[target_x, target_y, 0.0]], device=object_pos.device, dtype=object_pos.dtype,
    ).expand(object_pos.shape[0], -1)

    robot = env.scene["robot"]
    body_names = robot.data.body_names
    left_idx = _find_wrist_body_index(body_names, left_wrist_body, "left")
    right_idx = _find_wrist_body_index(body_names, right_wrist_body, "right")
    left_wrist_pos = robot.data.body_pos_w[:, left_idx, :3]
    right_wrist_pos = robot.data.body_pos_w[:, right_idx, :3]

    left_dist = torch.norm(left_wrist_pos - object_pos, dim=-1)
    right_dist = torch.norm(right_wrist_pos - object_pos, dim=-1)
    hand_to_object = torch.min(left_dist, right_dist)

    obj_height = object_pos[:, 2] - table_height
    obj_to_target_xy = torch.norm(object_pos[:, :2] - target_pos[:, :2], dim=-1)

    num_envs = object_pos.shape[0]
    reward = torch.zeros(num_envs, device=object_pos.device)

    # Tier 1 (0.2): hand near object
    reached = hand_to_object < reach_threshold
    reward = torch.where(reached, torch.tensor(0.2, device=reward.device), reward)

    # Tier 2 (0.4): object lifted off table
    grasped = obj_height > grasp_height_above_table
    reward = torch.where(grasped, torch.tensor(0.4, device=reward.device), reward)

    # Tier 3 (0.6): object lifted significantly
    lifted = obj_height > lift_height_above_table
    reward = torch.where(lifted, torch.tensor(0.6, device=reward.device), reward)

    # Tier 4 (0.8): object near target (XY) AND lifted
    near_target = (obj_to_target_xy < target_xy_threshold) & lifted
    reward = torch.where(near_target, torch.tensor(0.8, device=reward.device), reward)

    # Tier 5 (1.0): object on target, near table height, hand released
    on_target = obj_to_target_xy < target_xy_threshold
    near_table = obj_height < place_height_above_table
    hand_released = hand_to_object > release_distance
    placed = on_target & near_table & hand_released
    reward = torch.where(placed, torch.tensor(1.0, device=reward.device), reward)

    return reward
