"""Ghost Grasp Manager for Inspire Hand in Isaac Lab simulation.

The Inspire RH56DFQ hand has proximal-only joints and cannot physically grasp
objects in Isaac Lab's PhysX simulation. GhostGraspManager bypasses this by:
  1. Detecting when the hand is in a grasp pose (fingers closed + wrist near bottle)
  2. Recording the relative pose (offset) between wrist and bottle at grasp time
  3. Teleporting the bottle every sim step to follow the wrist (via write_root_pose_to_sim)
  4. Zeroing velocity to prevent PhysX force accumulation
  5. Releasing (reverting to physics) when fingers open near the target

Design principles:
  - All operations are GPU-compatible tensor ops (no USD API, no FixedJoint)
  - num_envs dimension throughout (vectorized for parallel Isaac Lab envs)
  - No external dependencies beyond torch
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper: rotation matrix → quaternion (w, x, y, z)  via Shepperd's method
# ---------------------------------------------------------------------------

def matrix_to_quat_wxyz(rot: Tensor) -> Tensor:
    """Convert batch of 3x3 rotation matrices to quaternions (w, x, y, z).

    Uses Shepperd's numerically stable method.

    Args:
        rot: [N, 3, 3] rotation matrices.

    Returns:
        [N, 4] quaternions in (w, x, y, z) order.
    """
    N = rot.shape[0]
    device = rot.device
    dtype = rot.dtype

    # Trace and diagonal elements
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]  # [N]

    quat = torch.zeros(N, 4, device=device, dtype=dtype)

    # Case 1: trace > 0
    mask0 = trace > 0.0
    if mask0.any():
        s = torch.sqrt(trace[mask0] + 1.0) * 2.0  # s = 4*w
        w = 0.25 * s
        x = (rot[mask0, 2, 1] - rot[mask0, 1, 2]) / s
        y = (rot[mask0, 0, 2] - rot[mask0, 2, 0]) / s
        z = (rot[mask0, 1, 0] - rot[mask0, 0, 1]) / s
        quat[mask0] = torch.stack([w, x, y, z], dim=-1)

    # Case 2: R[0,0] is largest diagonal
    mask1 = (~mask0) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    if mask1.any():
        s = torch.sqrt(1.0 + rot[mask1, 0, 0] - rot[mask1, 1, 1] - rot[mask1, 2, 2]) * 2.0
        w = (rot[mask1, 2, 1] - rot[mask1, 1, 2]) / s
        x = 0.25 * s
        y = (rot[mask1, 0, 1] + rot[mask1, 1, 0]) / s
        z = (rot[mask1, 0, 2] + rot[mask1, 2, 0]) / s
        quat[mask1] = torch.stack([w, x, y, z], dim=-1)

    # Case 3: R[1,1] is largest diagonal
    mask2 = (~mask0) & (~mask1) & (rot[:, 1, 1] > rot[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + rot[mask2, 1, 1] - rot[mask2, 0, 0] - rot[mask2, 2, 2]) * 2.0
        w = (rot[mask2, 0, 2] - rot[mask2, 2, 0]) / s
        x = (rot[mask2, 0, 1] + rot[mask2, 1, 0]) / s
        y = 0.25 * s
        z = (rot[mask2, 1, 2] + rot[mask2, 2, 1]) / s
        quat[mask2] = torch.stack([w, x, y, z], dim=-1)

    # Case 4: R[2,2] is largest diagonal
    mask3 = (~mask0) & (~mask1) & (~mask2)
    if mask3.any():
        s = torch.sqrt(1.0 + rot[mask3, 2, 2] - rot[mask3, 0, 0] - rot[mask3, 1, 1]) * 2.0
        w = (rot[mask3, 1, 0] - rot[mask3, 0, 1]) / s
        x = (rot[mask3, 0, 2] + rot[mask3, 2, 0]) / s
        y = (rot[mask3, 1, 2] + rot[mask3, 2, 1]) / s
        z = 0.25 * s
        quat[mask3] = torch.stack([w, x, y, z], dim=-1)

    return quat


# ---------------------------------------------------------------------------
# Helper: pose [N,7] (pos xyz + quat wxyz) → homogeneous [N,4,4]
# ---------------------------------------------------------------------------

def _pose_to_matrix(pose: Tensor) -> Tensor:
    """Convert [N,7] pose (xyz, quat wxyz) to [N,4,4] homogeneous transform."""
    N = pose.shape[0]
    device = pose.device
    dtype = pose.dtype

    pos = pose[:, :3]       # [N, 3]
    quat = pose[:, 3:]      # [N, 4] — (w, x, y, z)

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    # Build rotation matrix from quaternion
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

    mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(N, -1, -1).clone()
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = pos
    return mat


def _matrix_to_pose(mat: Tensor) -> Tensor:
    """Convert [N,4,4] homogeneous transform to [N,7] pose (xyz, quat wxyz)."""
    pos = mat[:, :3, 3]               # [N, 3]
    rot = mat[:, :3, :3]              # [N, 3, 3]
    quat = matrix_to_quat_wxyz(rot)   # [N, 4]
    return torch.cat([pos, quat], dim=-1)


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------

class GhostGraspManager:
    """Manages kinematic teleport attach/detach for Ghost Grasp.

    During grasp: teleport bottle to track wrist pose every step.
    During release: let physics take over.

    Example usage in the env step::

        target_pose, zero_vel, is_grasped = ghost.step(
            wrist_pose, bottle_pose, finger_mean,
            wrist_to_bottle_dist, bottle_to_target_xy
        )
        if is_grasped.any():
            bottle_asset.write_root_pose_to_sim(target_pose[is_grasped])
            bottle_asset.write_root_velocity_to_sim(zero_vel[is_grasped])

    Args:
        num_envs: Number of parallel simulation environments.
        device: Torch device (e.g. "cpu" or "cuda:0").
        grasp_finger_threshold: Finger mean angle (rad) above which a grasp is triggered.
        release_finger_threshold: Finger mean angle (rad) below which a release is triggered.
        grasp_wrist_distance: Max wrist-to-bottle distance (m) to trigger grasp.
        release_target_distance: Max bottle-to-target XY distance (m) to trigger release.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        grasp_finger_threshold: float = 1.2,
        release_finger_threshold: float = 0.3,
        grasp_wrist_distance: float = 0.05,
        release_target_distance: float = 0.03,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.grasp_finger_threshold = grasp_finger_threshold
        self.release_finger_threshold = release_finger_threshold
        self.grasp_wrist_distance = grasp_wrist_distance
        self.release_target_distance = release_target_distance

        # [num_envs] — whether each env currently has the bottle attached
        self.is_grasped: Tensor = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # [num_envs, 4, 4] — relative transform T_wrist_inv * T_bottle at grasp time
        self.grasp_offset: Tensor = torch.eye(4, device=self.device).unsqueeze(0).expand(
            num_envs, -1, -1
        ).clone()

        # Pre-allocated zero velocity buffer [num_envs, 6]
        self._zero_vel: Tensor = torch.zeros(num_envs, 6, device=self.device)

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Clear grasp state for given envs (or all if env_ids is None).

        Args:
            env_ids: 1-D integer tensor of env indices to reset.
                     If None, resets all environments.
        """
        if env_ids is None:
            self.is_grasped.fill_(False)
            self.grasp_offset.copy_(
                torch.eye(4, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
            )
        else:
            self.is_grasped[env_ids] = False
            self.grasp_offset[env_ids] = torch.eye(4, device=self.device)

    # ------------------------------------------------------------------
    # Trigger checks
    # ------------------------------------------------------------------

    def check_grasp_trigger(
        self,
        finger_mean: Tensor,
        wrist_to_bottle_dist: Tensor,
    ) -> Tensor:
        """Return bool mask of envs that should initiate a grasp this step.

        A grasp is triggered when:
          - finger_mean > grasp_finger_threshold  (fingers are closed)
          - wrist_to_bottle_dist < grasp_wrist_distance  (wrist near bottle)
          - not already grasped

        Args:
            finger_mean: [num_envs] mean finger joint angle (rad).
            wrist_to_bottle_dist: [num_envs] Euclidean distance wrist→bottle (m).

        Returns:
            [num_envs] bool tensor.
        """
        return (
            (finger_mean > self.grasp_finger_threshold)
            & (wrist_to_bottle_dist < self.grasp_wrist_distance)
            & (~self.is_grasped)
        )

    def check_release_trigger(
        self,
        finger_mean: Tensor,
        bottle_to_target_xy: Tensor,
    ) -> Tensor:
        """Return bool mask of envs that should release the bottle this step.

        A release is triggered when:
          - finger_mean < release_finger_threshold  (fingers are opening)
          - bottle_to_target_xy < release_target_distance  (bottle above target)
          - currently grasped

        Args:
            finger_mean: [num_envs] mean finger joint angle (rad).
            bottle_to_target_xy: [num_envs] XY-plane distance bottle→target (m).

        Returns:
            [num_envs] bool tensor.
        """
        return (
            (finger_mean < self.release_finger_threshold)
            & (bottle_to_target_xy < self.release_target_distance)
            & self.is_grasped
        )

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(
        self,
        env_ids: Tensor,
        wrist_pose: Tensor,
        bottle_pose: Tensor,
    ) -> None:
        """Record the grasp offset and mark envs as grasped.

        Computes T_offset = T_wrist^{-1} * T_bottle for each env in env_ids.

        Args:
            env_ids: 1-D integer tensor of env indices.
            wrist_pose: [num_envs, 7] wrist poses (xyz + quat wxyz).
            bottle_pose: [num_envs, 7] bottle poses (xyz + quat wxyz).
        """
        if env_ids.numel() == 0:
            return

        T_wrist = _pose_to_matrix(wrist_pose[env_ids])   # [K, 4, 4]
        T_bottle = _pose_to_matrix(bottle_pose[env_ids])  # [K, 4, 4]

        # Invert wrist transform: T^{-1} = [R^T | -R^T t ; 0 0 0 1]
        R = T_wrist[:, :3, :3]        # [K, 3, 3]
        t = T_wrist[:, :3, 3:]        # [K, 3, 1]
        R_inv = R.transpose(-1, -2)   # [K, 3, 3]
        t_inv = -torch.bmm(R_inv, t)  # [K, 3, 1]

        T_wrist_inv = torch.eye(4, device=self.device, dtype=wrist_pose.dtype).unsqueeze(0).expand(
            env_ids.numel(), -1, -1
        ).clone()
        T_wrist_inv[:, :3, :3] = R_inv
        T_wrist_inv[:, :3, 3] = t_inv[:, :, 0]

        offset = torch.bmm(T_wrist_inv, T_bottle)  # [K, 4, 4]

        self.grasp_offset[env_ids] = offset
        self.is_grasped[env_ids] = True

    def detach(self, env_ids: Tensor) -> None:
        """Release the bottle and clear grasp state for given envs.

        Args:
            env_ids: 1-D integer tensor of env indices.
        """
        if env_ids.numel() == 0:
            return
        self.is_grasped[env_ids] = False
        self.grasp_offset[env_ids] = torch.eye(4, device=self.device)

    # ------------------------------------------------------------------
    # Kinematic update
    # ------------------------------------------------------------------

    def compute_bottle_target(self, wrist_pose: Tensor) -> tuple[Tensor, Tensor]:
        """Compute where the bottle should be this step for all grasped envs.

        target = T_wrist * T_offset

        Args:
            wrist_pose: [num_envs, 7] wrist poses (xyz + quat wxyz).

        Returns:
            target_pose: [num_envs, 7] target bottle pose (xyz + quat wxyz).
            zero_vel:    [num_envs, 6] zeros for velocity write.
        """
        T_wrist = _pose_to_matrix(wrist_pose)                # [N, 4, 4]
        T_target = torch.bmm(T_wrist, self.grasp_offset)     # [N, 4, 4]
        target_pose = _matrix_to_pose(T_target)              # [N, 7]
        return target_pose, self._zero_vel

    # ------------------------------------------------------------------
    # Full step
    # ------------------------------------------------------------------

    def step(
        self,
        wrist_pose: Tensor,
        bottle_pose: Tensor,
        finger_mean: Tensor,
        wrist_to_bottle_dist: Tensor,
        bottle_to_target_xy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run a full ghost-grasp update step.

        Checks grasp and release triggers, updates attachment state, and
        computes the kinematic target pose for the bottle.

        Args:
            wrist_pose: [num_envs, 7] right wrist poses.
            bottle_pose: [num_envs, 7] current bottle poses.
            finger_mean: [num_envs] mean finger joint angle (rad).
            wrist_to_bottle_dist: [num_envs] wrist-to-bottle distance (m).
            bottle_to_target_xy: [num_envs] bottle-to-target XY distance (m).

        Returns:
            target_pose: [num_envs, 7] — desired bottle pose (only meaningful where is_grasped).
            zero_vel:    [num_envs, 6] — zeros to write as bottle velocity.
            is_grasped:  [num_envs] bool — which envs are currently grasped.
        """
        # --- grasp trigger ---
        grasp_mask = self.check_grasp_trigger(finger_mean, wrist_to_bottle_dist)
        grasp_ids = grasp_mask.nonzero(as_tuple=False).squeeze(-1)
        if grasp_ids.numel() > 0:
            self.attach(grasp_ids, wrist_pose, bottle_pose)

        # --- release trigger ---
        release_mask = self.check_release_trigger(finger_mean, bottle_to_target_xy)
        release_ids = release_mask.nonzero(as_tuple=False).squeeze(-1)
        if release_ids.numel() > 0:
            self.detach(release_ids)

        # --- kinematic update ---
        target_pose, zero_vel = self.compute_bottle_target(wrist_pose)

        return target_pose, zero_vel, self.is_grasped.clone()
