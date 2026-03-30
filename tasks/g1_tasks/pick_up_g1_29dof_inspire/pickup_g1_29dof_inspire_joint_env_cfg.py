# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Pick-up task with graduated reward for G1 + Inspire hand.

Uses 4-tier reward (reach/grasp/lift/place) instead of binary post-placement.
Scene reuses the standard table + red block setup.
"""
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_redblock import TableRedBlockSceneCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
import isaaclab.sim as sim_utils


@configclass
class PickUpSceneCfg(TableRedBlockSceneCfg):
    """Scene: table + red block + G1 Inspire hand.

    Object moved further from robot (Y=-4.20 vs default -4.03) to prevent
    hand-object contact at spawn while keeping within arm reach (~0.50m).
    """
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix(
        init_pos=(-4.2, -3.7, 0.76),
        init_rot=(0.7071, 0, 0, -0.7071),
    )
    # Override object position: 0.50m from robot (vs 0.33m default, 0.75m was too far)
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-4.25, -4.20, 0.84],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1),
            ),
        ),
    )
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera()


@configclass
class ActionsCfg:
    """Joint position control for arms + Inspire hands."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.reset_object_estimate)


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickUpG129InspireEnvCfg(ManagerBasedRLEnvCfg):
    """G1 pick-up environment with graduated reward and Inspire hand."""

    scene: PickUpSceneCfg = PickUpSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 2
        self.sim.physx.contact_offset = 0.015
        self.sim.physx.rest_offset = 0.001
        self.sim.physx.num_position_iterations = 12
        self.sim.physx.num_velocity_iterations = 4

        self.event_manager = SimpleEventManager()
        self.event_manager.register("reset_object_self", SimpleEvent(
            func=lambda env: base_mdp.reset_root_state_uniform(
                env,
                torch.arange(env.num_envs, device=env.device),
                pose_range={"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
                velocity_range={},
                asset_cfg=SceneEntityCfg("object"),
            )
        ))
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device),
            )
        ))
