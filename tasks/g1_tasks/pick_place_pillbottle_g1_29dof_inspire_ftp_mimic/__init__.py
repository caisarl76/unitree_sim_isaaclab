# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
import gymnasium as gym

gym.register(
    id="Isaac-PickPlace-PillBottle-G129-InspireFTP-Mimic",
    entry_point=f"{__name__}.pickplace_pillbottle_mimicgen_env_cfg:PickPlacePillBottleMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_pillbottle_mimicgen_env_cfg:PickPlacePillBottleMimicEnvCfg",
    },
    disable_env_checker=True,
)
