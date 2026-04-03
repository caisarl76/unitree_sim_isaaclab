"""Unit tests for 5-tier pick-and-place graduated reward.

Tests use mock env/scene objects to avoid Isaac Sim dependency.
"""
import sys
import types
import torch
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Stub out isaaclab and its sub-modules so the reward module can be imported
# without an Isaac Sim installation.
# ---------------------------------------------------------------------------
def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _pkg in [
    "isaaclab",
    "isaaclab.envs",
    "isaaclab.managers",
]:
    if _pkg not in sys.modules:
        _make_stub(_pkg)

# Minimal stubs that the reward module references at import time
sys.modules["isaaclab.envs"].ManagerBasedRLEnv = object
sys.modules["isaaclab.managers"].SceneEntityCfg = type(
    "SceneEntityCfg", (), {"__init__": lambda self, name="object": setattr(self, "name", name) or None}
)


def make_mock_env(
    object_pos,
    left_wrist_pos,
    right_wrist_pos,
    num_envs=1,
    body_names=None,
):
    """Create mock Isaac Lab env for reward function testing."""
    if body_names is None:
        body_names = [
            "pelvis", "left_hip_pitch_link", "left_hip_roll_link",
            "left_hip_yaw_link", "left_knee_link", "left_ankle_link",
            "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
            "right_knee_link", "right_ankle_link",
            "waist_yaw_link", "waist_roll_link",
            "left_shoulder_pitch_link", "left_shoulder_roll_link",
            "left_shoulder_yaw_link", "left_elbow_link",
            "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link",
            "right_shoulder_yaw_link", "right_elbow_link",
            "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
        ]

    object_pos_t = torch.tensor([object_pos], dtype=torch.float32)
    left_wrist_t = torch.tensor([[left_wrist_pos]], dtype=torch.float32)
    right_wrist_t = torch.tensor([[right_wrist_pos]], dtype=torch.float32)

    num_bodies = len(body_names)
    body_pos_w = torch.zeros(num_envs, num_bodies, 3)
    left_idx = body_names.index("left_wrist_yaw_link")
    right_idx = body_names.index("right_wrist_yaw_link")
    body_pos_w[0, left_idx] = left_wrist_t[0, 0]
    body_pos_w[0, right_idx] = right_wrist_t[0, 0]

    robot_data = SimpleNamespace(body_names=body_names, body_pos_w=body_pos_w)
    robot = SimpleNamespace(data=robot_data)

    object_data = SimpleNamespace(root_pos_w=object_pos_t)
    obj = SimpleNamespace(data=object_data)

    scene = {"robot": robot, "object": obj}

    env = MagicMock()
    env.scene.__getitem__ = lambda self, key: scene[key]
    env.num_envs = num_envs
    return env


class TestGraduatedReward5Tier:
    """Test each reward tier independently."""

    def setup_method(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "graduated_reward_pickplace_5tier",
            os.path.join(
                os.path.dirname(__file__),
                "..", "tasks", "common_rewards", "graduated_reward_pickplace_5tier.py"
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.compute_reward = mod.compute_reward

    def test_tier0_idle(self):
        """Hand far from object, object on table -> reward 0.0"""
        env = make_mock_env(
            object_pos=(-4.21, -3.864, 0.84),
            left_wrist_pos=(-4.2, -3.7, 0.9),
            right_wrist_pos=(-4.2, -3.7, 0.9),
        )
        reward = self.compute_reward(env)
        assert reward.item() == pytest.approx(0.0)

    def test_tier1_reach(self):
        """Hand near object (< 7cm) but object still on table -> reward 0.2"""
        env = make_mock_env(
            object_pos=(-4.21, -3.864, 0.84),
            left_wrist_pos=(-4.21, -3.864, 0.88),
            right_wrist_pos=(-4.2, -3.7, 0.9),
        )
        reward = self.compute_reward(env)
        assert reward.item() == pytest.approx(0.2)

    def test_tier2_grasp(self):
        """Object lifted 6cm above table -> reward 0.4"""
        env = make_mock_env(
            object_pos=(-4.21, -3.864, 0.88),
            left_wrist_pos=(-4.21, -3.864, 0.88),
            right_wrist_pos=(-4.2, -3.7, 0.9),
        )
        reward = self.compute_reward(env)
        assert reward.item() == pytest.approx(0.4)

    def test_tier3_lift(self):
        """Object lifted 10cm+ above table -> reward 0.6"""
        env = make_mock_env(
            object_pos=(-4.21, -3.864, 0.92),
            left_wrist_pos=(-4.21, -3.864, 0.92),
            right_wrist_pos=(-4.2, -3.7, 0.9),
        )
        reward = self.compute_reward(env)
        assert reward.item() == pytest.approx(0.6)

    def test_tier4_near_target(self):
        """Object within 5cm XY of target -> reward 0.8"""
        env = make_mock_env(
            object_pos=(-4.21, -3.69, 0.92),
            left_wrist_pos=(-4.21, -3.69, 0.92),
            right_wrist_pos=(-4.2, -3.7, 0.9),
        )
        reward = self.compute_reward(env)
        assert reward.item() == pytest.approx(0.8)

    def test_tier5_placed(self):
        """Object on target + low height (placed) + both hands released -> reward 1.0"""
        env = make_mock_env(
            object_pos=(-4.21, -3.664, 0.86),
            left_wrist_pos=(-4.21, -3.5, 0.95),
            right_wrist_pos=(-4.21, -3.85, 0.95),  # >10cm from object
        )
        reward = self.compute_reward(env)
        assert reward.item() == pytest.approx(1.0)

    def test_tier_ordering_highest_wins(self):
        """Higher tier always takes precedence."""
        env = make_mock_env(
            object_pos=(-4.21, -3.664, 0.92),
            left_wrist_pos=(-4.21, -3.664, 0.92),
            right_wrist_pos=(-4.2, -3.7, 0.9),
        )
        reward = self.compute_reward(env)
        assert reward.item() == pytest.approx(0.8)
