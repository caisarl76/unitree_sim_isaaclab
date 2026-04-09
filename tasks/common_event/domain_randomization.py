# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Domain randomization event functions for MimicGen data generation.

These functions are called during Isaac Lab environment resets to randomize
visual and physical properties, reducing the sim-to-real gap for the
generated MimicGen episodes.

Functions that require USD prim access (light intensity, camera pose) are
partially implemented here and marked with TODO for completion inside the
Isaac Lab container where omni.usd and pxr are available.

Usage (inside an IsaacLab EventCfg or manually in reset):
    from tasks.common_event.domain_randomization import (
        randomize_bottle_physics,
        randomize_arm_actuator_gains,
        randomize_light_color_temperature,
    )
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _uniform(low: float, high: float, shape: Tuple[int, ...], device: torch.device) -> Tensor:
    """Sample uniform random tensor in [low, high]."""
    return torch.empty(shape, device=device).uniform_(low, high)


def _kelvin_to_rgb(kelvin: float) -> Tuple[float, float, float]:
    """Convert a color temperature in Kelvin to a normalized (R, G, B) tuple.

    Algorithm adapted from Tanner Helland's approximation
    (https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html).

    Args:
        kelvin: Color temperature in Kelvin (e.g. 3000 – 6500).

    Returns:
        (r, g, b) each in [0.0, 1.0].
    """
    temp = kelvin / 100.0

    # --- Red ---
    if temp <= 66:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
        r = max(0.0, min(255.0, r))

    # --- Green ---
    if temp <= 66:
        g = 99.4708025861 * math.log(temp) - 161.1195681661
        g = max(0.0, min(255.0, g))
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
        g = max(0.0, min(255.0, g))

    # --- Blue ---
    if temp >= 66:
        b = 255.0
    elif temp <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(temp - 10) - 305.0447927307
        b = max(0.0, min(255.0, b))

    return r / 255.0, g / 255.0, b / 255.0


# ---------------------------------------------------------------------------
# Light randomization
# ---------------------------------------------------------------------------


def randomize_light_intensity(
    env,
    env_ids: Tensor,
    intensity_range: Tuple[float, float] = (0.3, 1.5),
) -> None:
    """Randomize ambient light intensity across environments.

    Requires USD prim access via omni.usd / pxr, which is only available
    inside the Isaac Lab container (Omniverse runtime). This function provides
    the correct call structure; the inner prim manipulation is marked as TODO.

    Args:
        env:             IsaacLab ManagerBasedRLEnv (or subclass).
        env_ids:         1-D int tensor of environment indices to randomize.
        intensity_range: (min, max) light intensity multipliers. Default spans
                         dim-office (0.3) to bright-studio (1.5).
    """
    n = env_ids.shape[0]
    intensities = _uniform(intensity_range[0], intensity_range[1], (n,), device=env.device)

    # TODO: Obtain the stage handle and iterate over light prims per env.
    #
    # Example skeleton (requires Omniverse runtime inside container):
    #
    #   import omni.usd
    #   from pxr import UsdLux
    #
    #   stage = omni.usd.get_context().get_stage()
    #   for idx, env_id in enumerate(env_ids.tolist()):
    #       # Each Isaac Lab env lives under /World/envs/env_{env_id}/
    #       light_prim_path = f"/World/envs/env_{env_id}/DistantLight"
    #       light = UsdLux.DistantLight(stage.GetPrimAtPath(light_prim_path))
    #       if light:
    #           light.GetIntensityAttr().Set(float(intensities[idx]) * 1000.0)
    #
    # Until then, log a debug message so callers know the function was invoked.
    if hasattr(env, "_debug_dr") and env._debug_dr:
        print(
            f"[DR] randomize_light_intensity: {n} envs, "
            f"range={intensity_range}  — TODO: prim access"
        )


def randomize_light_color_temperature(
    env,
    env_ids: Tensor,
    temp_range: Tuple[float, float] = (3000.0, 6500.0),
) -> None:
    """Randomize light color temperature across environments.

    Samples a color temperature uniformly from temp_range, converts to
    RGB using the Tanner Helland approximation (_kelvin_to_rgb), and applies
    it to the scene light prim.  The prim-write portion requires Omniverse
    runtime and is marked as TODO.

    Args:
        env:        IsaacLab env.
        env_ids:    1-D int tensor of environment indices.
        temp_range: (min_K, max_K) color temperature range in Kelvin.
                    3000 K ≈ warm incandescent, 6500 K ≈ daylight.
    """
    n = env_ids.shape[0]
    temps_cpu = _uniform(temp_range[0], temp_range[1], (n,), device=torch.device("cpu"))

    # Pre-compute RGB tuples for all sampled temperatures
    rgb_tuples = [_kelvin_to_rgb(float(t)) for t in temps_cpu]

    # TODO: Write RGB values to USD light prim per env.
    #
    # Example skeleton:
    #
    #   import omni.usd
    #   from pxr import UsdLux, Gf
    #
    #   stage = omni.usd.get_context().get_stage()
    #   for idx, env_id in enumerate(env_ids.tolist()):
    #       light_prim_path = f"/World/envs/env_{env_id}/DistantLight"
    #       light = UsdLux.DistantLight(stage.GetPrimAtPath(light_prim_path))
    #       if light:
    #           r, g, b = rgb_tuples[idx]
    #           light.GetColorAttr().Set(Gf.Vec3f(r, g, b))
    #
    if hasattr(env, "_debug_dr") and env._debug_dr:
        print(
            f"[DR] randomize_light_color_temperature: {n} envs, "
            f"range={temp_range} K — TODO: prim access"
        )


# ---------------------------------------------------------------------------
# Camera pose jitter
# ---------------------------------------------------------------------------


def randomize_camera_pose(
    env,
    env_ids: Tensor,
    pos_range: float = 0.01,
    rot_range_deg: float = 2.0,
) -> None:
    """Add small random jitter to the head camera extrinsics.

    Simulates imperfect camera mounting (±10 mm translation, ±2° rotation).
    Requires USD prim access to write the camera xform; marked as TODO.

    Args:
        env:          IsaacLab env.
        env_ids:      1-D int tensor of environment indices.
        pos_range:    Maximum translation jitter in metres (applied to each axis
                      independently, sampled from [-pos_range, pos_range]).
        rot_range_deg: Maximum rotation jitter in degrees per axis.
    """
    n = env_ids.shape[0]
    # Sample position deltas [N, 3]
    pos_deltas = _uniform(-pos_range, pos_range, (n, 3), device=env.device)
    # Sample Euler angle deltas [N, 3] in radians
    rot_range_rad = math.radians(rot_range_deg)
    rot_deltas = _uniform(-rot_range_rad, rot_range_rad, (n, 3), device=env.device)

    # TODO: Apply pos_deltas + rot_deltas to the camera USD prim per env.
    #
    # Example skeleton:
    #
    #   import omni.usd
    #   from pxr import UsdGeom, Gf
    #
    #   stage = omni.usd.get_context().get_stage()
    #   for idx, env_id in enumerate(env_ids.tolist()):
    #       cam_path = f"/World/envs/env_{env_id}/Robot/head_camera"
    #       xform = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
    #       if xform:
    #           ops = xform.GetOrderedXformOps()
    #           # Modify the translate op (index 0 by convention):
    #           translate_op = ops[0]
    #           base_t = translate_op.Get()
    #           dx, dy, dz = pos_deltas[idx].cpu().tolist()
    #           translate_op.Set(Gf.Vec3d(base_t[0]+dx, base_t[1]+dy, base_t[2]+dz))
    #           # Rotation modification depends on the xform op order in USD.
    #
    if hasattr(env, "_debug_dr") and env._debug_dr:
        print(
            f"[DR] randomize_camera_pose: {n} envs, "
            f"pos_range={pos_range}m, rot_range={rot_range_deg}° — TODO: prim access"
        )


# ---------------------------------------------------------------------------
# Object physics randomization
# ---------------------------------------------------------------------------


def randomize_bottle_physics(
    env,
    env_ids: Tensor,
    mass_range: Tuple[float, float] = (0.08, 0.15),
    static_friction_range: Tuple[float, float] = (0.5, 1.0),
    dynamic_friction_range: Tuple[float, float] = (0.3, 0.8),
) -> None:
    """Randomize pill bottle mass and friction coefficients via root_physx_view.

    This uses the Isaac Lab / Isaac Sim PhysX view API to write material
    properties directly.  The `env.scene["object"]` rigid body must expose a
    `root_physx_view` attribute (available from Isaac Lab 2.x onwards).

    Args:
        env:                    IsaacLab env with scene["object"] = bottle.
        env_ids:                1-D int tensor of environment indices.
        mass_range:             (min_kg, max_kg) bottle mass.  Typical PET pill
                                bottle: ~0.08–0.15 kg filled.
        static_friction_range:  (min, max) static friction coefficient.
        dynamic_friction_range: (min, max) dynamic friction coefficient.
    """
    n = env_ids.shape[0]
    masses = _uniform(mass_range[0], mass_range[1], (n,), device=env.device)
    s_frictions = _uniform(
        static_friction_range[0], static_friction_range[1], (n,), device=env.device
    )
    d_frictions = _uniform(
        dynamic_friction_range[0], dynamic_friction_range[1], (n,), device=env.device
    )

    try:
        bottle = env.scene["object"]
        physx_view = bottle.root_physx_view  # PhysX articulation/rigid body view

        # --- Mass ---
        # root_physx_view.get_masses() → [num_envs, 1] CPU tensor
        masses_all = physx_view.get_masses()  # [N_total, 1]
        masses_all[env_ids, 0] = masses.cpu()
        physx_view.set_masses(masses_all, indices=env_ids.cpu())

        # --- Friction ---
        # get_material_properties() → [num_envs, num_shapes, 3]
        # shape dim: [static_friction, dynamic_friction, restitution]
        mat_props = physx_view.get_material_properties()  # [N_total, S, 3]
        mat_props[env_ids, :, 0] = s_frictions.cpu().unsqueeze(-1)
        mat_props[env_ids, :, 1] = d_frictions.cpu().unsqueeze(-1)
        physx_view.set_material_properties(mat_props, indices=env_ids.cpu())

    except AttributeError as exc:
        # root_physx_view may not be available in older Isaac Lab versions.
        print(
            f"[DR] randomize_bottle_physics: root_physx_view not available "
            f"({exc}).  Skipping physics randomization."
        )
    except Exception as exc:
        print(f"[DR] randomize_bottle_physics: unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Actuator gain randomization
# ---------------------------------------------------------------------------


def randomize_arm_actuator_gains(
    env,
    env_ids: Tensor,
    stiffness_range: Tuple[float, float] = (2700.0, 3300.0),
    damping_range: Tuple[float, float] = (90.0, 110.0),
) -> None:
    """Randomize arm PD gains ±10% around nominal values.

    Nominal gains from the Unitree G1 URDF: stiffness ≈ 3000, damping ≈ 100.
    Range covers ±10% to simulate actuator wear and individual robot variation.

    Args:
        env:               IsaacLab env.
        env_ids:           1-D int tensor of environment indices.
        stiffness_range:   (min, max) position gain (Kp).  Default ±10% of 3000.
        damping_range:     (min, max) velocity gain (Kd).  Default ±10% of 100.

    TODO: The exact API for writing per-env actuator gains depends on the Isaac
    Lab version and actuator model used (IdealPDActuator vs DCMotor).  The
    approach below is correct for Isaac Lab 2.x with ImplicitActuator where
    `scene["robot"].actuators["arm"]` provides `stiffness` and `damping`
    tensors.  Verify the actuator name key ("arm", "left_arm", "right_arm")
    against the env config before enabling.
    """
    n = env_ids.shape[0]
    stiffness_vals = _uniform(
        stiffness_range[0], stiffness_range[1], (n,), device=env.device
    )
    damping_vals = _uniform(
        damping_range[0], damping_range[1], (n,), device=env.device
    )

    # TODO: Identify the correct actuator key in env.scene["robot"].actuators.
    #
    # Example skeleton for Isaac Lab 2.x:
    #
    #   robot = env.scene["robot"]
    #   # The actuator key may be "arm" or separate "left_arm"/"right_arm".
    #   # Check: list(robot.actuators.keys())
    #   actuator_key = "arm"
    #   if actuator_key in robot.actuators:
    #       act = robot.actuators[actuator_key]
    #       # act.stiffness: [N_total, num_joints_in_actuator]
    #       # act.damping:   [N_total, num_joints_in_actuator]
    #       act.stiffness[env_ids] = stiffness_vals.unsqueeze(-1)
    #       act.damping[env_ids]   = damping_vals.unsqueeze(-1)
    #   else:
    #       available = list(robot.actuators.keys())
    #       print(f"[DR] Actuator key '{actuator_key}' not found. Available: {available}")
    #
    if hasattr(env, "_debug_dr") and env._debug_dr:
        print(
            f"[DR] randomize_arm_actuator_gains: {n} envs, "
            f"K={stiffness_range}, D={damping_range} — TODO: verify actuator key"
        )
