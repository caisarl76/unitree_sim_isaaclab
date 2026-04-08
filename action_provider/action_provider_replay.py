# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from action_provider.action_base import ActionProvider
from typing import Optional
import torch
from tools.data_json_load import load_robot_data, load_robot_data_no_simstate, sim_state_to_json
from tools.shared_memory_utils import MultiImageReader
from tools.episode_writer import EpisodeWriter
import json
from typing import List, Optional
import numpy as np
import time
from tools.data_convert import convert_to_gripper_range
class FileActionProviderReplay(ActionProvider):
    """Action provider based on DDS"""
    
    def __init__(self,env, args_cli):
        super().__init__("FileActionProviderReplay")
        self.env = env
        self.enable_robot = args_cli.robot_type
        self.enable_gripper = args_cli.enable_dex1_dds
        self.enable_dex3 = args_cli.enable_dex3_dds
        self.enable_inspire = args_cli.enable_inspire_dds
        self.generate_data = args_cli.generate_data
        self.generate_data_dir = args_cli.generate_data_dir
        self.action_index = 10**1000
        self.total_step_num =0
        self.start_loop = True
        self.saved_data = True
        self.no_sim_state = False
        self.robot_state = []
        self.hand_state = []
        self.all_joint_names = env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}
        self._setup_joint_mapping()
        self.multi_image_reader=None
        self.recorder=None
        if self.generate_data:
            try:
                self.multi_image_reader = MultiImageReader()
                print(f"[{self.name}] MultiImageReader created")
            except Exception as e:
                print(f"[{self.name}] MultiImageReader creation failed: {e}")
                print(f"[{self.name}] Image data saving will be disabled")
                self.multi_image_reader = None
            
            self.recorder = EpisodeWriter(task_dir = self.generate_data_dir, frequency = 30, rerun_log = False)
        # Detect available cameras for image capture
        camera_keys = [k for k in env.scene.keys() if "camera" in k.lower()]
        self._camera_names = []
        if "front_camera" in camera_keys:
            self._camera_names.append("head")
        if "left_wrist_camera" in camera_keys:
            self._camera_names.append("left")
        if "right_wrist_camera" in camera_keys:
            self._camera_names.append("right")
        if not self._camera_names and camera_keys:
            self._camera_names.append("head")  # fallback: first camera as head
        self._image_count = len(self._camera_names)
        print(f"[{self.name}] Available cameras: {self._camera_names} (count={self._image_count})")

        print(f"FileActionProviderReplay init ok")


        device = self.env.device
        if hasattr(self, "left_arm_joint_indices"):
            self._left_arm_idx_t = torch.tensor(self.left_arm_joint_indices, dtype=torch.long, device=device)
        if hasattr(self, "right_arm_joint_indices"):
            self._right_arm_idx_t = torch.tensor(self.right_arm_joint_indices, dtype=torch.long, device=device)
        if hasattr(self, "left_hand_joint_indices"):
            self._left_hand_idx_t = torch.tensor(self.left_hand_joint_indices, dtype=torch.long, device=device)
        if hasattr(self, "right_hand_joint_indices"):
            self._right_hand_idx_t = torch.tensor(self.right_hand_joint_indices, dtype=torch.long, device=device)

    def load_data(self, file_path):
        """Load episode data. Falls back to no-sim-state mode for real-world recordings."""
        # Inspect first frame to decide which loader to use (avoids fragile exception matching)
        import json as _json
        with open(file_path, 'r') as f:
            content = _json.load(f)
        first_frame = (content.get("data") or [{}])[0]
        first_sim_state = first_frame.get("sim_state")
        has_sim_state = first_sim_state is not None and first_sim_state != {} and first_sim_state != ""

        if has_sim_state:
            self.robot_action, self.hand_action, self.sim_state_list, self.task_name_list, self.sim_state_json_list = load_robot_data(file_path)
            self.no_sim_state = False
        else:
            print(f"[{self.name}] No sim_state in data, using action-based replay")
            self.robot_action, self.hand_action, self.robot_state, self.hand_state, task_name = load_robot_data_no_simstate(file_path)
            self.no_sim_state = True
            self.sim_state_list = [None]
            self.task_name_list = [task_name]
            self.sim_state_json_list = [None]

        self.total_step_num = len(self.robot_action)
        self.total_hand_step_num = len(self.hand_action)
        if self.generate_data:
            self.recorder.create_episode()
            self.saved_data = False

        self.start_loop = False

        if self.no_sim_state:
            return None, self.task_name_list[0]
        return self.sim_state_list[0], self.task_name_list[0]
    def start_replay(self):
        self.action_index=0
    def get_start_loop(self):
        return self.start_loop
    def _setup_joint_mapping(self):
        """Setup joint mapping"""
        if self.enable_robot == "g129" or self.enable_robot == "h1_2":
            self.arm_joint_mapping = {
                "left_shoulder_pitch_joint": 0,
                "left_shoulder_roll_joint": 1,
                "left_shoulder_yaw_joint": 2,
                "left_elbow_joint": 3,
                "left_wrist_roll_joint": 4,
                "left_wrist_pitch_joint": 5,
                "left_wrist_yaw_joint": 6,
                "right_shoulder_pitch_joint": 7,
                "right_shoulder_roll_joint": 8,
                "right_shoulder_yaw_joint": 9,
                "right_elbow_joint": 10,
                "right_wrist_roll_joint": 11,
                "right_wrist_pitch_joint": 12,
                "right_wrist_yaw_joint": 13
            }
            self.left_arm_joint = [        
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint"]
            self.right_arm_joint = [        
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint"]
            self.left_arm_joint_indices = [self.joint_to_index[name] for name in self.left_arm_joint]
            self.right_arm_joint_indices = [self.joint_to_index[name] for name in self.right_arm_joint]
            
        if self.enable_gripper:
            self.gripper_joint_mapping = {
                "left_hand_Joint1_1": 1,
                "left_hand_Joint2_1": 1,
                "right_hand_Joint1_1": 0,
                "right_hand_Joint2_1": 0,
            }
            self.left_hand_joint = ["left_hand_Joint1_1"]
            self.right_hand_joint = ["right_hand_Joint1_1"]
        if self.enable_dex3:
            self.left_hand_joint_mapping = {
                "left_hand_thumb_0_joint":0,
                "left_hand_thumb_1_joint":1,
                "left_hand_thumb_2_joint":2,
                "left_hand_middle_0_joint":3,
                "left_hand_middle_1_joint":4,
                "left_hand_index_0_joint":5,
                "left_hand_index_1_joint":6}
            self.right_hand_joint_mapping = {
                "right_hand_thumb_0_joint":0,     
                "right_hand_thumb_1_joint":1,
                "right_hand_thumb_2_joint":2,
                "right_hand_middle_0_joint":3,
                "right_hand_middle_1_joint":4,
                "right_hand_index_0_joint":5,
                "right_hand_index_1_joint":6}
            self.left_hand_joint = [
                # hand joints (14)
                # left hand (7)
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
                "left_hand_index_0_joint",
                "left_hand_index_1_joint"]
            self.right_hand_joint = [
                    "right_hand_thumb_0_joint",
                    "right_hand_thumb_1_joint",
                    "right_hand_thumb_2_joint",
                    "right_hand_middle_0_joint",
                    "right_hand_middle_1_joint",
                    "right_hand_index_0_joint",
                    "right_hand_index_1_joint",
                ]
        if self.enable_inspire:
            self.left_hand_joint = [
                "L_pinky_proximal_joint",
                "L_ring_proximal_joint",
                "L_middle_proximal_joint",
                "L_index_proximal_joint",
                "L_thumb_proximal_pitch_joint",
                "L_thumb_proximal_yaw_joint",
            ]
            self.right_hand_joint = [
                "R_pinky_proximal_joint",
                "R_ring_proximal_joint",
                "R_middle_proximal_joint",
                "R_index_proximal_joint",
                "R_thumb_proximal_pitch_joint",
                "R_thumb_proximal_yaw_joint",
            ]
        self.left_hand_joint_indices = [self.joint_to_index[name] for name in self.left_hand_joint]
        self.right_hand_joint_indices = [self.joint_to_index[name] for name in self.right_hand_joint]
        self.all_joint_indices = self.left_arm_joint_indices + self.right_arm_joint_indices #+ self.left_hand_joint_indices + self.right_hand_joint_indices


    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from DDS"""
        try:
            # Get robot command
            if self.action_index < self.total_step_num:
                if self.enable_robot == "g129" or self.enable_robot == "h1_2":
                    arm_cmd_data = self.robot_action[self.action_index]

                # Get gripper command
                if self.enable_gripper:
                    hand_cmd_data = self.hand_action[self.action_index]

                # Get hand command
                elif self.enable_dex3:
                    hand_cmd_data = self.hand_action[self.action_index]
                elif self.enable_inspire:
                    hand_cmd_data = self.hand_action[self.action_index]

                if self.no_sim_state:
                    # Action-based replay: set joint positions directly from recorded states
                    self._set_joints_from_state(env, self.action_index)
                    env.sim.render()

                    if self.generate_data:
                        # Capture current sim state for output
                        try:
                            env_state = env.scene.get_state()
                            env_state_json = sim_state_to_json(env_state)
                            captured_sim_state = {"init_state": env_state_json, "task_name": self.task_name_list[0]}
                        except Exception:
                            captured_sim_state = None
                        # Save frame data (read camera directly from sensor data)
                        self._save_frame_direct(env, arm_cmd_data, hand_cmd_data, captured_sim_state)
                else:
                    # Original sim_state replay: force scene to recorded state
                    env.scene.reset_to(self.sim_state_list[self.action_index], torch.tensor([0], device=env.device), is_relative=True)

                    if self.generate_data:
                        for sensor in env.scene.sensors.values():
                            sensor.update(0.02, force_recompute=False)
                        env.sim.render()
                        env.observation_manager.compute()
                        while not self.save_date(env, arm_cmd_data, hand_cmd_data, self.sim_state_json_list[self.action_index]):
                            time.sleep(0.01)
                            env.sim.render()
                            env.observation_manager.compute()
                    else:
                        env.sim.render()

                self.action_index += 1
            else:
                self.action_index = 10**1000
                if self.generate_data:
                    if not self.saved_data:
                        self.recorder.save_episode()
                        self.saved_data = True
                    if self.recorder.is_available:
                        self.start_loop=True
                else:
                    self.start_loop=True

            return None

        except Exception as e:
            print(f"[{self.name}] Get DDS action failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _set_joints_from_state(self, env, idx):
        """Set robot joint positions directly from recorded real-world states.

        Uses write_joint_state_to_sim() to bypass the action space and directly
        place the robot at the recorded configuration, then steps physics once.
        """
        robot = env.scene["robot"]
        device = env.device

        joint_pos = robot.data.joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.joint_vel)

        arm_state = self.robot_state[idx]   # [left_arm(7), right_arm(7)]
        hand_state = self.hand_state[idx]   # [right_hand(N), left_hand(N)]

        # Set arm joint positions
        joint_pos[0, self._left_arm_idx_t] = torch.tensor(
            arm_state[:7], dtype=torch.float32, device=device
        )
        joint_pos[0, self._right_arm_idx_t] = torch.tensor(
            arm_state[7:], dtype=torch.float32, device=device
        )

        # Set hand joint positions
        # hand_state layout: [right_hand(N), left_hand(N)] — matches load_robot_data convention
        n_hand = len(self.right_hand_joint_indices)

        joint_pos[0, self._right_hand_idx_t] = torch.tensor(
            hand_state[:n_hand], dtype=torch.float32, device=device
        )
        joint_pos[0, self._left_hand_idx_t] = torch.tensor(
            hand_state[n_hand:], dtype=torch.float32, device=device
        )

        # Write to simulation and step physics
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        env.sim.step(render=False)
    

    def cleanup(self):
        """Clean up DDS resources"""
        if self.multi_image_reader:
            self.multi_image_reader.close()
        if self.recorder:
            self.recorder.close()
        self.is_running = False
        print(f"[{self.name}] Resource cleanup completed")
    def get_state(self,env):

        joint_pos = env.scene["robot"].data.joint_pos
        left_arm_joint_pose = joint_pos[:,self.left_arm_joint_indices][0].detach().cpu().numpy().tolist()
        right_arm_joint_pose = joint_pos[:,self.right_arm_joint_indices][0].detach().cpu().numpy().tolist()
        if self.enable_gripper:
            left_hand_joint_pose = np.array(convert_to_gripper_range(joint_pos[:,self.left_hand_joint_indices][0].detach().cpu().numpy())).tolist()
            right_hand_joint_pose = np.array(convert_to_gripper_range(joint_pos[:,self.right_hand_joint_indices][0].detach().cpu().numpy())).tolist()
        else:
            left_hand_joint_pose = joint_pos[:,self.left_hand_joint_indices][0].detach().cpu().numpy().tolist()
            right_hand_joint_pose = joint_pos[:,self.right_hand_joint_indices][0].detach().cpu().numpy().tolist()


        return left_arm_joint_pose,right_arm_joint_pose,left_hand_joint_pose,right_hand_joint_pose

    def get_images(self, image_count=None):
        """Get images using read_single_image for each available camera."""
        if image_count is None:
            image_count = self._image_count
        images = {}
        names = self._camera_names[:image_count] if hasattr(self, '_camera_names') else ['head', 'left', 'right'][:image_count]

        for name in names:
            image = self.multi_image_reader.read_single_image(name)
            if image is not None:
                images[name] = image

        if len(images) < 1:
            return None

        return images

    def _save_frame_direct(self, env, arm_action, hand_action, sim_state=None):
        """Save frame by reading camera images directly from sensors (no shared memory)."""
        def ensure_list(data):
            if isinstance(data, list):
                return data
            elif hasattr(data, 'tolist'):
                return data.tolist()
            elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                return list(data)
            else:
                return [data]

        left_arm_state, right_arm_state, left_ee_state, right_ee_state = self.get_state(env)

        # Read camera images directly from sensors
        colors = {}
        camera_map = {"front_camera": 0, "left_wrist_camera": 1, "right_wrist_camera": 2}
        for cam_name, idx in camera_map.items():
            if cam_name in [k for k in env.scene.keys()]:
                try:
                    img = env.scene[cam_name].data.output["rgb"][0]
                    if img.device.type != 'cpu':
                        img = img.cpu()
                    colors[f"color_{idx}"] = img.numpy()
                except Exception as e:
                    print(f"[save_frame_direct] Failed to read {cam_name}: {e}")

        if not colors:
            return  # no camera data, skip frame

        # Capture depth if available (e.g., when using RGBD camera config)
        depths = {}
        for cam_name, idx in camera_map.items():
            if cam_name in [k for k in env.scene.keys()]:
                try:
                    cam_output = env.scene[cam_name].data.output
                    if "distance_to_camera" in cam_output:
                        depth_data = cam_output["distance_to_camera"][0]
                        if depth_data.device.type != 'cpu':
                            depth_data = depth_data.cpu()
                        # Convert to uint16 mm for storage compatibility
                        depth_mm = (depth_data.numpy() * 1000).clip(0, 65535).astype(np.uint16)
                        depths[f"depth_{idx}"] = depth_mm
                except Exception as e:
                    print(f"[save_frame_direct] Failed to read depth from {cam_name}: {e}")

        left_arm_action = arm_action[:7].tolist()
        right_arm_action = arm_action[7:].tolist()
        if self.enable_gripper:
            right_hand_action = np.array(hand_action[:1]).tolist()
            left_hand_action = np.array(hand_action[1:]).tolist()
        elif self.enable_dex3:
            right_hand_action = hand_action[:7].tolist()
            left_hand_action = hand_action[7:].tolist()
        elif self.enable_inspire:
            right_hand_action = hand_action[:6].tolist()
            left_hand_action = hand_action[6:].tolist()
        else:
            right_hand_action = []
            left_hand_action = []

        states = {
            "left_arm": {"qpos": ensure_list(left_arm_state), "qvel": [], "torque": []},
            "right_arm": {"qpos": ensure_list(right_arm_state), "qvel": [], "torque": []},
            "left_ee": {"qpos": ensure_list(left_ee_state), "qvel": [], "torque": []},
            "right_ee": {"qpos": ensure_list(right_ee_state), "qvel": [], "torque": []},
            "body": {"qpos": []},
        }
        actions = {
            "left_arm": {"qpos": ensure_list(left_arm_action), "qvel": [], "torque": []},
            "right_arm": {"qpos": ensure_list(right_arm_action), "qvel": [], "torque": []},
            "left_ee": {"qpos": ensure_list(left_hand_action), "qvel": [], "torque": []},
            "right_ee": {"qpos": ensure_list(right_hand_action), "qvel": [], "torque": []},
        }

        self.recorder.add_item(
            colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state
        )

    def save_date(self,env,arm_action,hand_action,sim_state=None):
        def ensure_list(data):
            """Ensure data is list type, if not, convert to list"""
            if isinstance(data, list):
                return data
            elif hasattr(data, 'tolist'):  # numpy array or torch tensor
                return data.tolist()
            elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):  # other iterable types
                return list(data)
            else:  # single value
                return [data]
        
        left_arm_state,right_arm_state,left_ee_state,right_ee_state = self.get_state(env)
        images = self.get_images()
        if images is None:
            return False
        colors = {}
        depths = {}
        left_arm_action = arm_action[:7].tolist()
        right_arm_action = arm_action[7:].tolist()
        if self.enable_gripper:
            right_hand_action = np.array(hand_action[:1]).tolist() 
            left_hand_action = np.array(hand_action[1:]).tolist()
        elif self.enable_dex3:
            right_hand_action = hand_action[:7].tolist()
            left_hand_action = hand_action[7:].tolist()
        elif self.enable_inspire:
            right_hand_action = hand_action[:6].tolist()
            left_hand_action = hand_action[6:].tolist()
        for i, name in enumerate(["head", "left", "right"]):
            if name in images:
                colors[f"color_{i}"] = images[name]
        states = {
            "left_arm": {                                                                    
                "qpos":   ensure_list(left_arm_state),    
                "qvel":   [],                          
                "torque": [],                        
            }, 
            "right_arm": {                                                                    
                "qpos":   ensure_list(right_arm_state),       
                "qvel":   [],                          
                "torque": [],                         
            },                        
            "left_ee": {                                                                    
                "qpos":   ensure_list(left_ee_state),           
                "qvel":   [],                           
                "torque": [],                          
            }, 
            "right_ee": {                                                                    
                "qpos":   ensure_list(right_ee_state),       
                "qvel":   [],                           
                "torque": [],  
            }, 
            "body": {
                "qpos": [],
            }, 
        }
        actions = {
            "left_arm": {                                   
                "qpos":   ensure_list(left_arm_action),       
                "qvel":   [],       
                "torque": [],      
            }, 
            "right_arm": {                                   
                "qpos":   ensure_list(right_arm_action),       
                "qvel":   [],       
                "torque": [],       
            },                         
            "left_ee": {                                   
                "qpos":   ensure_list(left_hand_action),       
                "qvel":   [],       
                "torque": [],       
            }, 
            "right_ee": {                                   
                "qpos":   ensure_list(right_hand_action),       
                "qvel":   [],       
                "torque": [], 
            }, 
            "body": {
                "qpos": [],
            }, 
        }
        self.recorder.add_item(colors=colors, depths=depths, states=states, actions=actions,sim_state=sim_state)
        return True
    def sim_state_to_json(self,data):
        data_serializable = self.tensors_to_list(data)
        json_str = json.dumps(data_serializable)
        return json_str
    def tensors_to_list(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.tensors_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.tensors_to_list(i) for i in obj]
        return obj