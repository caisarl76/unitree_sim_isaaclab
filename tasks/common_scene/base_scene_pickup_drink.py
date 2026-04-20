# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
public base scene configuration module
provides reusable scene element configurations for drink bottle pick-up task.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg  # isort: skip
import os

project_root = os.environ.get("PROJECT_ROOT")


@configclass
class TableDrinkSceneCfg(InteractiveSceneCfg):
    """Drink bottle pick-up scene configuration class.

    Defines a scene containing robot, cylindrical drink bottle, table, and room.
    The drink is a cylinder (~6cm diameter x 15cm tall, ~300g).
    """

    # 1. room wall configuration
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

    # 2. table configuration
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

    # 3. drink bottle object configuration
    # Cylinder: ~6cm diameter x 15cm tall, ~300g
    # Spawn height: table surface ~0.81m + half object height 0.075m = 0.885m
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-4.25, -4.03, 0.885],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,   # 3cm radius = 6cm diameter
            height=0.15,   # 15cm tall
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.5, 0.8),  # blue-ish drink bottle
                metallic=0.2,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=6.0,
                dynamic_friction=1.0,
                restitution=0.01,
            ),
        ),
    )

    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-4.1, -4.9, 1.8),
        rot_offset=(-0.3173, 0.94833, 0.0, 0.0),
    )
