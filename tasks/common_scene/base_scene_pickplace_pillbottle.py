# repos/unitree_sim_isaaclab/tasks/common_scene/base_scene_pickplace_pillbottle.py
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Scene config for pill bottle pick-and-place with red tape target area."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg
import os

project_root = os.environ.get("PROJECT_ROOT")


@configclass
class TablePillBottleSceneCfg(InteractiveSceneCfg):
    """Table scene with pill bottle object and red tape target area.

    Pill bottle: cylinder ~10cm tall, ~4cm diameter, 40g (empty plastic bottle).
    Red tape: 15cm x 15cm flat area marking the place target.
    Table and warehouse from the standard redblock scene.
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

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-4.3, -4.2, -0.2],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",
        ),
    )

    # Pill bottle (cylinder approximation)
    # Position: same as redblock default, adjusted per compute_object_positions.py
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-4.21, -3.864, 0.84],  # PillBottle position from GT wrist centroids
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.02,   # 4cm diameter
            height=0.10,   # 10cm tall
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),  # 40g empty bottle
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

    # Red tape target area (static, non-graspable)
    target_area = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TargetArea",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-4.21, -3.664, 0.811],  # ~20cm in front of bottle (toward robot)
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.002),  # 15cm x 15cm, very thin
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1), metallic=0.0,  # Red
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    # World camera (overview)
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-4.1, -4.9, 1.8),
        rot_offset=(-0.3173, 0.94833, 0.0, 0.0),
    )
