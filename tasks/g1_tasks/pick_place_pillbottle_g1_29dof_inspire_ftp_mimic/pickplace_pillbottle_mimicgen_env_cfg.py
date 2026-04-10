# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""MimicGen environment config for pill bottle pick-and-place with Ghost Grasp.

This module provides:
  - PickPlacePillBottleMimicEnvCfg: @configclass extending the FTP parent env.
  - PickPlacePillBottleMimicEnv:    ManagerBasedRLMimicEnv subclass that wires
    Ghost Grasp, subtask signals, and Placo IK into the MimicGen interface.

The class follows the ManagerBasedRLMimicEnv interface:
  get_robot_eef_pose()             → [N, 4, 4]
  get_object_poses()               → dict[str, [N, 4, 4]]
  get_subtask_term_signals()       → dict[str, Tensor[bool]]
  target_eef_pose_to_action()      → [N, 26]
  actions_to_target_eef_pose()     → [N, 4, 4]
  get_datagen_info()               → dict

Ghost Grasp runs inside _post_physics_step() to kinematically attach the
bottle to the wrist when fingers are closed and the wrist is near the bottle.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from isaaclab.utils import configclass

from ..pick_place_pillbottle_g1_29dof_inspire_ftp.pickplace_pillbottle_g1_29dof_inspire_ftp_joint_env_cfg import (
    PickPlacePillBottleG129InspireFTPEnvCfg,
)

from .ghost_grasp import GhostGraspManager, matrix_to_quat_wxyz
from .subtask_signals import (
    compute_subtask_signals,
    compute_finger_mean,
    compute_wrist_to_bottle_dist,
    compute_bottle_to_target_xy,
)
from .ik_solver import PlacoIKSolver, compose_26d_action, compose_53d_action, RIGHT_ARM_ACTION_INDICES

# ---------------------------------------------------------------------------
# Try to import ManagerBasedRLMimicEnv (available when isaaclab_mimic installed)
# ---------------------------------------------------------------------------
try:
    from isaaclab.envs import ManagerBasedRLMimicEnv
    from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
    _BASE_CLS = ManagerBasedRLMimicEnv
    _HAS_MIMIC = True
except ImportError:
    from isaaclab.envs import ManagerBasedRLEnv
    _BASE_CLS = ManagerBasedRLEnv
    _HAS_MIMIC = False


# ---------------------------------------------------------------------------
# Subtask config constant
# ---------------------------------------------------------------------------

SUBTASK_CONFIGS = [
    {"name": "reach",     "type": "free_space",  "term_offset_range": (10, 20)},
    {"name": "grasp",     "type": "on_object",   "object": "bottle", "term_offset_range": (5, 10)},
    {"name": "lift",      "type": "on_object",   "object": "bottle", "term_offset_range": (5, 10)},
    {"name": "transport", "type": "free_space",  "term_offset_range": (10, 20)},
    {"name": "place",     "type": "on_object",   "object": "target", "term_offset_range": (0, 0)},
]


# ---------------------------------------------------------------------------
# Helper: [N, 4] quat wxyz → [N, 3, 3] rotation matrix
# ---------------------------------------------------------------------------

def quat_wxyz_to_matrix(quat: Tensor) -> Tensor:
    """Convert batch of quaternions (w, x, y, z) to 3x3 rotation matrices.

    Args:
        quat: [N, 4] quaternions in (w, x, y, z) order.

    Returns:
        [N, 3, 3] rotation matrices.
    """
    N = quat.shape[0]
    device = quat.device
    dtype = quat.dtype

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    rot = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def _pose7_to_mat4(pose: Tensor) -> Tensor:
    """Convert [N, 7] pose (xyz + quat wxyz) to [N, 4, 4] homogeneous matrix."""
    N = pose.shape[0]
    device = pose.device
    dtype = pose.dtype

    pos = pose[:, :3]
    rot = quat_wxyz_to_matrix(pose[:, 3:])  # [N, 3, 3]

    mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(N, -1, -1).clone()
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = pos
    return mat


# ---------------------------------------------------------------------------
# Inspire hand joint indices within the full articulation state
# ---------------------------------------------------------------------------

# Right-hand Inspire joint indices (12 joints) inside the robot articulation.
# Order: [R_pinky_proximal, R_ring_proximal, R_middle_proximal, R_index_proximal,
#         R_thumb_bend,      R_index_distal,  R_little_proximal,  R_little_distal,
#         R_ring_distal,     R_thumb_ab,      R_thumb_proximal,   R_middle_distal]
INSPIRE_RIGHT_HAND_JOINT_INDICES = [36, 37, 35, 34, 48, 38, 31, 32, 30, 29, 43, 33]


# ---------------------------------------------------------------------------
# Env config dataclass
# ---------------------------------------------------------------------------

if _HAS_MIMIC:
    @configclass
    class PickPlacePillBottleMimicEnvCfg(PickPlacePillBottleG129InspireFTPEnvCfg, MimicEnvCfg):
        """MimicGen env config — inherits scene/reward/obs from FTP parent + MimicGen datagen."""

        def __post_init__(self):
            super().__post_init__()

            # MimicGen datagen settings
            self.datagen_config.name = "demo_src_pickplace_pillbottle_g1_inspire"
            self.datagen_config.generation_guarantee = True
            self.datagen_config.generation_keep_failed = True
            self.datagen_config.generation_num_trials = 10
            self.datagen_config.generation_select_src_per_subtask = True
            self.datagen_config.generation_transform_first_robot_pose = False
            self.datagen_config.generation_interpolate_from_last_target_pose = True
            self.datagen_config.generation_relative = True
            self.datagen_config.max_num_failures = 25
            self.datagen_config.seed = 1

            # 5-phase subtask decomposition for pick-and-place
            subtask_configs = []
            subtask_configs.append(SubTaskConfig(
                object_ref="object",  # pill bottle rigid body name in scene
                subtask_term_signal="reach",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Reach to pill bottle",
                next_subtask_description="Grasp pill bottle",
            ))
            subtask_configs.append(SubTaskConfig(
                object_ref="object",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp pill bottle",
                next_subtask_description="Lift pill bottle",
            ))
            subtask_configs.append(SubTaskConfig(
                object_ref="object",
                subtask_term_signal="lift",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Lift pill bottle",
                next_subtask_description="Transport to target",
            ))
            subtask_configs.append(SubTaskConfig(
                object_ref="object",
                subtask_term_signal="transport",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Transport to target zone",
                next_subtask_description="Place pill bottle",
            ))
            subtask_configs.append(SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,  # final subtask
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place pill bottle on target",
            ))
            self.subtask_configs["robot"] = subtask_configs
else:
    @configclass
    class PickPlacePillBottleMimicEnvCfg(PickPlacePillBottleG129InspireFTPEnvCfg):
        """Fallback env config when isaaclab_mimic is not installed."""
        pass


# ---------------------------------------------------------------------------
# No-camera variant (avoids Isaac Lab 2.3.2 camera init bug in headless)
# ---------------------------------------------------------------------------

from tasks.common_config import G1RobotPresets
from tasks.common_scene.base_scene_pickplace_pillbottle import TablePillBottleSceneCfg
from isaaclab.assets import ArticulationCfg

@configclass
class _PillBottleFTPNoCamSceneCfg(TablePillBottleSceneCfg):
    """Pill bottle scene WITHOUT camera (avoids Isaac Lab 2.3.2 headless bug)."""
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_ftp_base_fix(
        init_pos=(-4.2, -3.7, 0.76),
        init_rot=(0.7071, 0, 0, -0.7071),
    )
    # NO front_camera


if _HAS_MIMIC:
    @configclass
    class PickPlacePillBottleMimicNoCamEnvCfg(PickPlacePillBottleMimicEnvCfg):
        """MimicGen env without camera — for annotation and headless generation."""

        def __post_init__(self):
            super().__post_init__()
            # Replace scene with no-camera variant, preserving spacing
            nocam_scene = _PillBottleFTPNoCamSceneCfg(
                num_envs=self.scene.num_envs,
                env_spacing=self.scene.env_spacing,
            )
            self.scene = nocam_scene
            # Disable camera obs (set to None — del breaks configclass)
            self.observations.policy.camera_image = None
else:
    @configclass
    class PickPlacePillBottleMimicNoCamEnvCfg(PickPlacePillBottleG129InspireFTPEnvCfg):
        """Fallback no-camera env config."""

        def __post_init__(self):
            super().__post_init__()
            nocam_scene = _PillBottleFTPNoCamSceneCfg(
                num_envs=self.scene.num_envs,
                env_spacing=self.scene.env_spacing,
            )
            self.scene = nocam_scene
            self.observations.policy.camera_image = None


# ---------------------------------------------------------------------------
# Env class
# ---------------------------------------------------------------------------

class PickPlacePillBottleMimicEnv(_BASE_CLS):
    """ManagerBasedRLMimicEnv subclass for pill bottle pick-and-place.

    Integrates:
      - Ghost Grasp (kinematic teleport for Inspire hand in PhysX)
      - Subtask signals (5-phase: reach/grasp/lift/transport/place)
      - Placo IK (right-arm target pose → joint positions)
    """

    cfg: PickPlacePillBottleMimicEnvCfg

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, cfg: PickPlacePillBottleMimicEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Ghost Grasp manager
        self.ghost_grasp = GhostGraspManager(
            num_envs=self.num_envs,
            device=self.device,
        )

        # Right wrist body index
        body_names = self.scene["robot"].data.body_names
        self._right_wrist_idx: int = body_names.index("right_wrist_yaw_link")

        # Target position from reward params
        reward_params = self.cfg.rewards.reward.params
        target_x: float = reward_params["target_x"]
        target_y: float = reward_params["target_y"]
        # z is table surface height (roughly 0.81 m)
        self._target_pos = torch.tensor(
            [target_x, target_y, 0.81],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(self.num_envs, -1)  # [N, 3]

        # IK solver (lazy — created on first call to avoid import overhead at
        # import time when placo may not be installed)
        self._ik_solver: PlacoIKSolver | None = None

    def _get_ik_solver(self) -> PlacoIKSolver:
        """Lazy-initialise PlacoIKSolver on first call."""
        if self._ik_solver is None:
            from pathlib import Path
            # Assume URDF lives under PROJECT_ROOT/assets/robots/
            import os
            project_root = os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
            urdf_path = Path(project_root) / "assets" / "robots" / "g1_29dof_inspire_ftp.urdf"
            self._ik_solver = PlacoIKSolver(urdf_path=urdf_path)
        return self._ik_solver

    # ------------------------------------------------------------------
    # Post-physics Ghost Grasp step
    # ------------------------------------------------------------------

    def _post_physics_step(self) -> None:
        """Run Ghost Grasp kinematic teleport after every physics step."""
        super()._post_physics_step()

        robot = self.scene["robot"]
        bottle = self.scene["object"]

        # --- wrist pose [N, 7] (xyz + quat wxyz) ---
        # body_state_w: [N, num_bodies, 13] — pos(3)+quat(4)+linvel(3)+angvel(3)
        body_state = robot.data.body_state_w  # [N, B, 13]
        wrist_state = body_state[:, self._right_wrist_idx, :]  # [N, 13]
        wrist_pose = wrist_state[:, :7]  # [N, 7] pos + quat wxyz

        # --- bottle pose [N, 7] ---
        bottle_pose = bottle.data.root_state_w[:, :7]  # [N, 7]

        # --- finger mean [N] ---
        joint_pos = robot.data.joint_pos  # [N, num_joints]
        finger_pos = joint_pos[:, INSPIRE_RIGHT_HAND_JOINT_INDICES]  # [N, 12]
        finger_mean = compute_finger_mean(finger_pos)  # [N]

        # --- distances ---
        wrist_to_bottle = compute_wrist_to_bottle_dist(
            wrist_pose[:, :3], bottle_pose[:, :3]
        )  # [N]
        bottle_to_target = compute_bottle_to_target_xy(
            bottle_pose[:, :3], self._target_pos
        )  # [N]

        # --- ghost grasp step ---
        target_pose, zero_vel, is_grasped = self.ghost_grasp.step(
            wrist_pose=wrist_pose,
            bottle_pose=bottle_pose,
            finger_mean=finger_mean,
            wrist_to_bottle_dist=wrist_to_bottle,
            bottle_to_target_xy=bottle_to_target,
        )

        # --- teleport grasped envs ---
        if is_grasped.any():
            grasped_ids = is_grasped.nonzero(as_tuple=False).squeeze(-1)
            bottle.write_root_pose_to_sim(target_pose[grasped_ids], env_ids=grasped_ids)
            bottle.write_root_velocity_to_sim(zero_vel[grasped_ids], env_ids=grasped_ids)

    # ------------------------------------------------------------------
    # Reset hook
    # ------------------------------------------------------------------

    def _on_reset(self, env_ids: Tensor) -> None:
        """Reset Ghost Grasp state for the given environments."""
        super()._on_reset(env_ids)
        self.ghost_grasp.reset(env_ids=env_ids)

    # ------------------------------------------------------------------
    # MimicGen interface
    # ------------------------------------------------------------------

    def get_robot_eef_pose(self, eef_name: str = "robot", env_ids: Sequence[int] | None = None) -> Tensor:
        """Return right wrist world pose as [N, 4, 4] homogeneous matrices.

        Args:
            eef_name: Name of the end effector (unused, single-arm task).
            env_ids: Environment indices. If None, all envs.

        Returns:
            [N, 4, 4] float32 tensors on self.device.
        """
        body_state = self.scene["robot"].data.body_state_w  # [N, B, 13]
        wrist_state = body_state[:, self._right_wrist_idx, :]  # [N, 13]
        wrist_pose = wrist_state[:, :7]  # [N, 7] xyz + quat wxyz
        return _pose7_to_mat4(wrist_pose)  # [N, 4, 4]

    def get_object_poses(self) -> dict[str, Tensor]:
        """Return object poses as [N, 4, 4] homogeneous matrices.

        Returns:
            Dict with key "bottle" → [N, 4, 4].
        """
        bottle_pose = self.scene["object"].data.root_state_w[:, :7]  # [N, 7]
        return {"bottle": _pose7_to_mat4(bottle_pose)}  # [N, 4, 4]

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, Tensor]:
        """Return five bool tensors indicating subtask completion.

        Args:
            env_ids: Optional subset of environment indices.

        Returns:
            Dict with keys "reach", "grasp", "lift", "transport", "place".
            Each value is a bool tensor of shape [len(env_ids)] or [N].
        """
        robot = self.scene["robot"]
        bottle_pose = self.scene["object"].data.root_state_w[:, :7]

        body_state = robot.data.body_state_w
        wrist_pos = body_state[:, self._right_wrist_idx, :3]  # [N, 3]
        bottle_pos = bottle_pose[:, :3]  # [N, 3]

        joint_pos = robot.data.joint_pos
        finger_pos = joint_pos[:, INSPIRE_RIGHT_HAND_JOINT_INDICES]  # [N, 12]

        return compute_subtask_signals(
            wrist_pos=wrist_pos,
            bottle_pos=bottle_pos,
            target_pos=self._target_pos,
            finger_positions=finger_pos,
            is_grasped=self.ghost_grasp.is_grasped,
            env_ids=env_ids,
        )

    def target_eef_pose_to_action(
        self,
        target_pose: Tensor,
        finger_actions: Tensor,
    ) -> Tensor:
        """Convert target EEF pose + finger actions to a 53-D action vector.

        Uses Placo IK to solve for right-arm joint positions. All other joints
        (legs, waist, left arm, left hand) stay at 0 (use_default_offset=True
        holds them at init pose). Operates per-env with CPU IK (Phase 1).

        Args:
            target_pose:    [N, 4, 4] desired right wrist pose in world frame.
            finger_actions: [N, 12] right hand joint position targets
                            (full 12-joint Inspire hand).

        Returns:
            [N, 53] action vectors for the full articulation.
        """
        N = target_pose.shape[0]
        ik = self._get_ik_solver()

        joint_pos_all = self.scene["robot"].data.joint_pos  # [N, num_joints]

        # Build output on CPU, then move to device
        actions_np = np.zeros((N, 53), dtype=np.float32)

        target_pose_np = target_pose.cpu().numpy()
        finger_np = finger_actions.cpu().numpy()

        for i in range(N):
            if i == 0:
                # Build the right-arm joint index mapping once
                if not hasattr(self, "_right_arm_joint_indices"):
                    from .ik_solver import RIGHT_ARM_JOINT_NAMES
                    all_joint_names = self.scene["robot"].data.joint_names
                    self._right_arm_joint_indices = [
                        all_joint_names.index(name) for name in RIGHT_ARM_JOINT_NAMES
                    ]

            right_arm_current = joint_pos_all[i, self._right_arm_joint_indices].cpu().numpy()

            # Solve IK
            right_arm_target = ik.solve(
                target_pose_4x4=target_pose_np[i],
                current_joint_pos=right_arm_current,
            )  # [7]

            actions_np[i] = compose_53d_action(
                right_arm=right_arm_target,
                right_hand=finger_np[i],
            )

        return torch.from_numpy(actions_np).to(device=self.device)

    def actions_to_target_eef_pose(self, actions: Tensor) -> Tensor:
        """Infer the current EEF pose from the given actions.

        MimicGen calls this to understand where the robot ended up.
        We delegate to get_robot_eef_pose() since actions alone cannot give
        the exact pose without FK (and the current state already reflects the
        latest action execution).

        Args:
            actions: [N, 53] action vectors (ignored).

        Returns:
            [N, 4, 4] current right wrist pose.
        """
        return self.get_robot_eef_pose()

    def get_datagen_info(self) -> dict:
        """Return metadata dict for MimicGen dataset generation.

        Returns:
            Dict with subtask configs and environment identifiers.
        """
        return {
            "env_id": "Isaac-PickPlace-PillBottle-G129-InspireFTP-Mimic",
            "subtask_configs": SUBTASK_CONFIGS,
            "action_dim": 53,
            "eef_name": "right_wrist_yaw_link",
            "objects": ["bottle"],
            "target_pos": self._target_pos[0].cpu().tolist(),
        }
