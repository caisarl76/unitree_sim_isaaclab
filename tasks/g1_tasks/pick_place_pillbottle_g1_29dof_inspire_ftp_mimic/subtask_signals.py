"""Subtask signal computation for MimicGen pick-and-place pipeline.

Computes five boolean signals that indicate when each subtask phase is complete.
These are used by ManagerBasedRLMimicEnv.get_subtask_term_signals() to segment
demonstrations into subtask boundaries for MimicGen data augmentation.

All operations are vectorized GPU tensor ops (no Python loops over envs).
Uses PyTorch & / | operators — NOT Python and / or — for element-wise boolean logic.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Inspire RH56DFQ right-hand finger joint indices within the 12-element
# per-hand joint vector:  [R_pinky, R_ring, R_middle, R_index]
RIGHT_FINGER_INDICES = [0, 1, 2, 3]

# Nominal table surface height in world frame (meters)
TABLE_HEIGHT = 0.81


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_finger_mean(finger_positions: Tensor) -> Tensor:
    """Compute mean position of right-hand proximal fingers.

    Args:
        finger_positions: [N, 12] inspire hand joint positions.

    Returns:
        [N] mean of joints at RIGHT_FINGER_INDICES (0, 1, 2, 3).
    """
    return finger_positions[:, RIGHT_FINGER_INDICES].mean(dim=1)


def compute_wrist_to_bottle_dist(wrist_pos: Tensor, bottle_pos: Tensor) -> Tensor:
    """Compute Euclidean distance between wrist and bottle (3D).

    Args:
        wrist_pos:  [N, 3] right wrist world position.
        bottle_pos: [N, 3] bottle world position.

    Returns:
        [N] Euclidean distance.
    """
    return torch.norm(wrist_pos - bottle_pos, dim=1)


def compute_bottle_to_target_xy(bottle_pos: Tensor, target_pos: Tensor) -> Tensor:
    """Compute XY-plane (horizontal) distance from bottle to target.

    Args:
        bottle_pos: [N, 3] bottle world position.
        target_pos: [N, 3] target world position.

    Returns:
        [N] distance in the XY plane only.
    """
    diff_xy = bottle_pos[:, :2] - target_pos[:, :2]
    return torch.norm(diff_xy, dim=1)


# ---------------------------------------------------------------------------
# Main signal computation
# ---------------------------------------------------------------------------

def compute_subtask_signals(
    wrist_pos: Tensor,
    bottle_pos: Tensor,
    target_pos: Tensor,
    finger_positions: Tensor,
    is_grasped: Tensor,
    env_ids: Sequence[int] | None = None,
) -> dict[str, Tensor]:
    """Compute five boolean subtask completion signals for all (or selected) envs.

    Args:
        wrist_pos:        [N, 3] right wrist world position.
        bottle_pos:       [N, 3] bottle world position.
        target_pos:       [N, 3] target place position.
        finger_positions: [N, 12] inspire hand joint positions.
        is_grasped:       [N] bool — True when GhostGraspManager has attached bottle.
        env_ids:          Optional subset of environment indices to compute for.
                          If provided, all inputs are sliced to this subset.

    Returns:
        Dict with keys "reach", "grasp", "lift", "transport", "place", each a
        bool tensor of shape [len(env_ids)] (or [N] if env_ids is None).
    """
    if env_ids is not None:
        idx = list(env_ids)
        wrist_pos = wrist_pos[idx]
        bottle_pos = bottle_pos[idx]
        target_pos = target_pos[idx]
        finger_positions = finger_positions[idx]
        is_grasped = is_grasped[idx]

    # Derived scalars — all shape [N]
    wrist_to_bottle = compute_wrist_to_bottle_dist(wrist_pos, bottle_pos)
    finger_mean = compute_finger_mean(finger_positions)
    bottle_z = bottle_pos[:, 2]
    wrist_to_target_xy = compute_bottle_to_target_xy(wrist_pos, target_pos)
    table_z = wrist_pos.new_tensor(TABLE_HEIGHT)

    # Five subtask signals
    reach = wrist_to_bottle < 0.07
    grasp = (finger_mean > 1.2) & is_grasped
    lift = bottle_z > (table_z + 0.10)
    transport = wrist_to_target_xy < 0.05
    place = (bottle_z < (table_z + 0.02)) & (finger_mean < 0.3)

    return {
        "reach": reach,
        "grasp": grasp,
        "lift": lift,
        "transport": transport,
        "place": place,
    }
