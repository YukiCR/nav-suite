# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, env_cfg_base, go1_env_cfg

gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Anymal-D-DEV",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavTasksDepthNavEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)
gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Anymal-D-TRAIN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavTasksDepthNavEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfg",
    },
)
gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Anymal-D-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavTasksDepthNavEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)

gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Go1-DEV",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.Go1NavTasksDepthNavEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)
gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Go1-TRAIN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.Go1NavTasksDepthNavEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfg",
    },
)
gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Go1-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.Go1NavTasksDepthNavEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)


gym.register(
    id="NavTasks-DepthImgNavigation-Go1-IL-COLLECT",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.Go1NavTasksDepthNavEnvCfg_IL_COLLECT,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfg",
    },
)

gym.register(
    id="NavTasks-DepthImgNavigation-Go1-IL-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.Go1NavTasksDepthNavEnvCfg_IL_PLAY,
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
)
