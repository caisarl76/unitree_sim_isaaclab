"""Placo IK solver wrapper for MimicGen pick-and-place pipeline.

Wraps the existing placo-based RobotKinematics class from unitree_IL_lerobot
for use in MimicGen's target_eef_pose_to_action() calls.

Phase 1 is CPU / single-env only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_INIT = np.zeros(7, dtype=np.float32)
LEFT_HAND_INIT = np.zeros(6, dtype=np.float32)


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def _import_robot_kinematics():
    """Lazily import RobotKinematics, trying multiple import paths.

    Returns:
        RobotKinematics class.

    Raises:
        ImportError: if neither placo nor the kinematics module is available.
    """
    # Primary: standard package import (works when unitree_lerobot is installed)
    try:
        from lerobot.model.kinematics import RobotKinematics  # type: ignore[import]
        return RobotKinematics
    except ImportError:
        pass

    # Fallback: inject the src path directly and retry
    _src_path = (
        Path(__file__).resolve().parents[5]
        / "unitree_IL_lerobot"
        / "unitree_lerobot"
        / "lerobot"
        / "src"
    )
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

    try:
        from lerobot.model.kinematics import RobotKinematics  # type: ignore[import]
        return RobotKinematics
    except ImportError as exc:
        raise ImportError(
            "Could not import RobotKinematics. "
            "Make sure placo is installed and unitree_IL_lerobot is available. "
            f"Tried sys.path with: {_src_path}"
        ) from exc


# ---------------------------------------------------------------------------
# PlacoIKSolver
# ---------------------------------------------------------------------------

class PlacoIKSolver:
    """Thin wrapper around RobotKinematics for right-arm IK in MimicGen.

    Uses radians internally (unlike RobotKinematics which uses degrees in its
    public API).  All public methods accept and return radians so that
    callers can pass joint positions from Isaac Lab directly.

    Args:
        urdf_path: Path to the robot URDF file.
        ee_frame: Name of the end-effector frame in the URDF (default: "R_ee").
    """

    def __init__(self, urdf_path: str | Path, ee_frame: str = "R_ee") -> None:
        RobotKinematics = _import_robot_kinematics()
        self._kin = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name=ee_frame,
            joint_names=RIGHT_ARM_JOINT_NAMES,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        target_pose_4x4: np.ndarray,
        current_joint_pos: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """Compute IK for the right arm.

        Args:
            target_pose_4x4: [4, 4] desired end-effector pose in world frame.
            current_joint_pos: [7] current right-arm joint positions (radians).
            position_weight: Weight for the position constraint.
            orientation_weight: Weight for the orientation constraint.

        Returns:
            [7] joint positions in radians that achieve the target pose.
        """
        target_pose_4x4 = np.asarray(target_pose_4x4, dtype=np.float64)
        if target_pose_4x4.shape != (4, 4):
            raise ValueError(f"target_pose_4x4 must be shape (4,4), got {target_pose_4x4.shape}")

        current_joint_pos = np.asarray(current_joint_pos, dtype=np.float64)

        # RobotKinematics works in degrees
        current_deg = np.rad2deg(current_joint_pos)
        result_deg = self._kin.inverse_kinematics(
            current_joint_pos=current_deg,
            desired_ee_pose=target_pose_4x4,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
        # inverse_kinematics may return more values than joint_names if
        # current_joint_pos was longer; slice to 7 and convert back.
        return np.deg2rad(result_deg[: len(RIGHT_ARM_JOINT_NAMES)]).astype(np.float32)

    def forward(self, joint_pos: np.ndarray) -> np.ndarray:
        """Compute FK for the right arm.

        Args:
            joint_pos: [7] right-arm joint positions in radians.

        Returns:
            [4, 4] end-effector pose in world frame.
        """
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        joint_deg = np.rad2deg(joint_pos)
        T = self._kin.forward_kinematics(joint_deg)
        return np.asarray(T, dtype=np.float32)


# ---------------------------------------------------------------------------
# compose_26d_action
# ---------------------------------------------------------------------------

def compose_26d_action(
    right_arm: np.ndarray,
    right_hand: np.ndarray,
    left_arm: np.ndarray | None = None,
    left_hand: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble a 26-D action vector matching the eval pipeline ordering.

    Ordering: [left_arm(7), right_arm(7), left_hand(6), right_hand(6)]

    Args:
        right_arm:  [7] right arm joint positions.
        right_hand: [6] right hand joint positions.
        left_arm:   [7] left arm joint positions; defaults to LEFT_ARM_INIT
                    (all zeros) when None.
        left_hand:  [6] left hand joint positions; defaults to LEFT_HAND_INIT
                    (all zeros) when None.

    Returns:
        [26] action vector.
    """
    right_arm = np.asarray(right_arm, dtype=np.float32)
    right_hand = np.asarray(right_hand, dtype=np.float32)

    if left_arm is None:
        left_arm = LEFT_ARM_INIT.copy()
    else:
        left_arm = np.asarray(left_arm, dtype=np.float32)

    if left_hand is None:
        left_hand = LEFT_HAND_INIT.copy()
    else:
        left_hand = np.asarray(left_hand, dtype=np.float32)

    if right_arm.shape != (7,):
        raise ValueError(f"right_arm must be shape (7,), got {right_arm.shape}")
    if right_hand.shape != (6,):
        raise ValueError(f"right_hand must be shape (6,), got {right_hand.shape}")
    if left_arm.shape != (7,):
        raise ValueError(f"left_arm must be shape (7,), got {left_arm.shape}")
    if left_hand.shape != (6,):
        raise ValueError(f"left_hand must be shape (6,), got {left_hand.shape}")

    return np.concatenate([left_arm, right_arm, left_hand, right_hand])
