"""ToastedBread pick-and-place scene configuration.

Same as RedBlock scene but with a bread-colored rectangular cuboid instead of a red cube.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg
import os

project_root = os.environ.get("PROJECT_ROOT")


@configclass
class TableToastedBreadSceneCfg(InteractiveSceneCfg):
    """Scene with table and a toasted bread object."""

    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",
        ),
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-4.3, -4.2, -0.2],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",
        ),
    )

    # ToastedBread — rectangular cuboid, golden brown color
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-4.25, -4.03, 0.84],
            rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.10, 0.04),  # bread slice: 12cm x 10cm x 4cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),  # ~300g bread
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.62, 0.35), metallic=0  # golden brown toast
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )

    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-4.1, -4.9, 1.8),
        rot_offset=(-0.3173, 0.94833, 0.0, 0.0)
    )
