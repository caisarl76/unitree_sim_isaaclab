# repos/unitree_sim_isaaclab/tasks/common_scene/base_scene_pickplace_pillbottle.py
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Scene config for pill bottle pick-and-place with red tape target area.

Uses programmatic table (no yellowbox USD) for clean visuals matching real setup.
"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg
import os

project_root = os.environ.get("PROJECT_ROOT")

# Table geometry constants
_TABLE_CENTER_X = -4.25
_TABLE_CENTER_Y = -4.15  # 10cm further from robot (was -4.05, too close)
_TABLE_SURFACE_Z = 0.81     # absolute Z of table surface top
_TABLE_WIDTH = 0.80          # X extent (m)
_TABLE_DEPTH = 0.60          # Y extent (m)
_TABLE_THICKNESS = 0.03      # surface slab thickness
_TABLE_LEG_HEIGHT = _TABLE_SURFACE_Z - _TABLE_THICKNESS / 2  # floor to underside
_TABLE_LEG_SIDE = 0.04       # leg cross-section

# Red tape outline constants (matching yellow crate footprint)
_TAPE_W = 0.02               # tape width 2cm
_BOX_X = 0.10                # target area 10x10cm (was 20x15cm, too large)
_BOX_Y = 0.10
_TARGET_X = -4.25            # target center X
_TARGET_Y = -4.00            # target center Y (was -3.92, too close to default wrist)
_TARGET_Z = _TABLE_SURFACE_Z + 0.001  # just above table
_RED_MAT = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.05, 0.05), metallic=0.0)
_NO_COLLIDE = sim_utils.CollisionPropertiesCfg(collision_enabled=False)


@configclass
class TablePillBottleSceneCfg(InteractiveSceneCfg):
    """Table scene with pill bottle object and red tape target area.

    Pill bottle: cylinder ~10cm tall, ~4cm diameter, 40g (empty plastic bottle).
    Red tape: 20cm x 20cm flat area marking the place target.
    Table built from primitives — no yellow box.
    """

    # Room
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",
        ),
    )

    # ── Table (programmatic: surface + 4 legs) ──────────────────────────
    # Surface slab — collision enabled so objects rest on it
    table_surface = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableSurface",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TABLE_CENTER_X, _TABLE_CENTER_Y, _TABLE_SURFACE_Z - _TABLE_THICKNESS / 2],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_WIDTH, _TABLE_DEPTH, _TABLE_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.35, 0.20), metallic=0.0,  # Wood brown
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Table legs (static, visual + collision)
    table_leg_fl = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLegFL",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TABLE_CENTER_X - _TABLE_WIDTH / 2 + _TABLE_LEG_SIDE,
                 _TABLE_CENTER_Y + _TABLE_DEPTH / 2 - _TABLE_LEG_SIDE,
                 _TABLE_LEG_HEIGHT / 2],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIDE, _TABLE_LEG_SIDE, _TABLE_LEG_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.28, 0.15), metallic=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_fr = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLegFR",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TABLE_CENTER_X + _TABLE_WIDTH / 2 - _TABLE_LEG_SIDE,
                 _TABLE_CENTER_Y + _TABLE_DEPTH / 2 - _TABLE_LEG_SIDE,
                 _TABLE_LEG_HEIGHT / 2],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIDE, _TABLE_LEG_SIDE, _TABLE_LEG_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.28, 0.15), metallic=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_bl = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLegBL",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TABLE_CENTER_X - _TABLE_WIDTH / 2 + _TABLE_LEG_SIDE,
                 _TABLE_CENTER_Y - _TABLE_DEPTH / 2 + _TABLE_LEG_SIDE,
                 _TABLE_LEG_HEIGHT / 2],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIDE, _TABLE_LEG_SIDE, _TABLE_LEG_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.28, 0.15), metallic=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    table_leg_br = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableLegBR",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TABLE_CENTER_X + _TABLE_WIDTH / 2 - _TABLE_LEG_SIDE,
                 _TABLE_CENTER_Y - _TABLE_DEPTH / 2 + _TABLE_LEG_SIDE,
                 _TABLE_LEG_HEIGHT / 2],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIDE, _TABLE_LEG_SIDE, _TABLE_LEG_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.28, 0.15), metallic=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # ── Pill bottle (cylinder approximation) ────────────────────────────
    # Position: on table, ~35cm from robot (matching RedBlock distance)
    # Robot at (-4.2, -3.7), facing -Y toward table
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-4.25, -4.15, 0.87],  # on table surface — 10cm further from robot
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.02,   # 4cm diameter
            height=0.10,   # 10cm tall
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),  # 100g — resist accidental pushes
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.95, 0.95), metallic=0.1,  # White plastic
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.05,
            ),
        ),
    )

    # ── Red tape target outline (4 strips, not filled) ─────────────────
    # Same footprint as the original yellow crate (~20x15cm), red tape border
    # Positioned between robot and bottle, ~22cm from robot on table

    # Top strip (along X, +Y edge)
    tape_top = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TapeTop",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TARGET_X, _TARGET_Y + _BOX_Y / 2 - _TAPE_W / 2, _TARGET_Z]),
        spawn=sim_utils.CuboidCfg(
            size=(_BOX_X, _TAPE_W, 0.003),
            visual_material=_RED_MAT, collision_props=_NO_COLLIDE),
    )
    # Bottom strip (along X, -Y edge)
    tape_bottom = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TapeBottom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TARGET_X, _TARGET_Y - _BOX_Y / 2 + _TAPE_W / 2, _TARGET_Z]),
        spawn=sim_utils.CuboidCfg(
            size=(_BOX_X, _TAPE_W, 0.003),
            visual_material=_RED_MAT, collision_props=_NO_COLLIDE),
    )
    # Left strip (along Y, -X edge)
    tape_left = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TapeLeft",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TARGET_X - _BOX_X / 2 + _TAPE_W / 2, _TARGET_Y, _TARGET_Z]),
        spawn=sim_utils.CuboidCfg(
            size=(_TAPE_W, _BOX_Y, 0.003),
            visual_material=_RED_MAT, collision_props=_NO_COLLIDE),
    )
    # Right strip (along Y, +X edge)
    tape_right = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TapeRight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_TARGET_X + _BOX_X / 2 - _TAPE_W / 2, _TARGET_Y, _TARGET_Z]),
        spawn=sim_utils.CuboidCfg(
            size=(_TAPE_W, _BOX_Y, 0.003),
            visual_material=_RED_MAT, collision_props=_NO_COLLIDE),
    )

    # World camera (overview)
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-4.1, -4.9, 1.8),
        rot_offset=(-0.3173, 0.94833, 0.0, 0.0),
    )


def randomize_table_material(env, env_ids, color_range=None):
    """Randomize table surface color at reset for domain randomization.

    Args:
        env: Isaac Lab environment
        env_ids: environment indices to randomize
        color_range: dict with 'r', 'g', 'b' each as [min, max] float.
            Default: wood tones from light beige to dark brown.
    """
    import omni.isaac.core.utils.prims as prim_utils

    if color_range is None:
        color_range = {
            'r': [0.30, 0.75],
            'g': [0.20, 0.55],
            'b': [0.10, 0.40],
        }

    for idx in env_ids:
        r = torch.empty(1).uniform_(color_range['r'][0], color_range['r'][1]).item()
        g = torch.empty(1).uniform_(color_range['g'][0], color_range['g'][1]).item()
        b = torch.empty(1).uniform_(color_range['b'][0], color_range['b'][1]).item()

        prim_path = f"/World/envs/env_{idx}/TableSurface"
        try:
            from pxr import UsdShade, Sdf
            stage = prim_utils.get_current_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                shader = UsdShade.Material.Get(stage, f"{prim_path}/Looks/PreviewSurface/Shader")
                if shader:
                    shader.GetInput("diffuseColor").Set((r, g, b))
        except Exception:
            pass  # Graceful degrade if Omni API unavailable


def randomize_bottle_material(env, env_ids, color_range=None):
    """Randomize bottle color at reset."""
    import omni.isaac.core.utils.prims as prim_utils

    if color_range is None:
        color_range = {
            'r': [0.70, 1.00],
            'g': [0.60, 1.00],
            'b': [0.50, 1.00],
        }

    for idx in env_ids:
        r = torch.empty(1).uniform_(color_range['r'][0], color_range['r'][1]).item()
        g = torch.empty(1).uniform_(color_range['g'][0], color_range['g'][1]).item()
        b = torch.empty(1).uniform_(color_range['b'][0], color_range['b'][1]).item()

        prim_path = f"/World/envs/env_{idx}/Object"
        try:
            from pxr import UsdShade
            stage = prim_utils.get_current_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                shader = UsdShade.Material.Get(stage, f"{prim_path}/Looks/PreviewSurface/Shader")
                if shader:
                    shader.GetInput("diffuseColor").Set((r, g, b))
        except Exception:
            pass
