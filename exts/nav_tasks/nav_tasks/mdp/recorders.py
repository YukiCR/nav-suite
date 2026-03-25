

from __future__ import annotations

import torch
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.utils import configclass

# Import built-in recorder cfgs to extend them
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    InitialStateRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
    PostStepProcessedActionsRecorderCfg,
)


class PreStepNavigationPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records navigation policy observations as a dict for robomimic.

    The observation dict contains:
    - base_lin_vel: 3 dims
    - base_ang_vel: 3 dims
    - joint_pos: 12 dims (Go1 has 12 joints)
    - joint_vel: 12 dims
    - goal_commands: 4 dims (x, y, z, yaw goal)
    - forwards_depth_image: 256 dims (embedded depth image from PerceptNet)

    Total: 290 dims

    This recorder unwraps the flat policy observation buffer into a dictionary format
    that is compatible with robomimic's expected observation structure.
    """

    def record_pre_step(self):
        # Get the flat policy observation buffer
        # Shape: [num_envs, 290]
        flat_obs = self._env.obs_buf["policy"]

        # Unwrap into dict format for robomimic
        obs_dict = {
            "base_lin_vel": flat_obs[..., 0:3],
            "base_ang_vel": flat_obs[..., 3:6],
            "joint_pos": flat_obs[..., 6:18],
            "joint_vel": flat_obs[..., 18:30],
            "goal_commands": flat_obs[..., 30:34],  # 4 dims, not 3!
            "forwards_depth_image": flat_obs[..., 34:290],
        }

        return "obs", obs_dict


@configclass
class PreStepNavigationPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for PreStepNavigationPolicyObservationsRecorder."""

    class_type: type[RecorderTerm] = PreStepNavigationPolicyObservationsRecorder


class PreStepNavigationAndLocomotionObservationsRecorder(RecorderTerm):
    """Recorder term that records both navigation and locomotion policy observations.

    This is useful when you want to record all observation groups for analysis or
    when the IL policy needs access to both high-level navigation and low-level
    locomotion observations.

    Navigation policy observations (290 dims):
    - base_lin_vel: 3 dims
    - base_ang_vel: 3 dims
    - joint_pos: 12 dims
    - joint_vel: 12 dims
    - goal_commands: 4 dims
    - forwards_depth_image: 256 dims

    Locomotion policy observations (48 dims for Go1 without height scan):
    - base_lin_vel: 3 dims
    - base_ang_vel: 3 dims
    - projected_gravity: 3 dims
    - velocity_commands: 3 dims
    - joint_pos: 12 dims
    - joint_vel: 12 dims
    - actions: 12 dims
    """

    def record_pre_step(self):
        obs_dict = {}

        # Navigation policy observations
        if "policy" in self._env.obs_buf:
            nav_obs = self._env.obs_buf["policy"]
            obs_dict["nav_policy"] = {
                "base_lin_vel": nav_obs[..., 0:3],
                "base_ang_vel": nav_obs[..., 3:6],
                "joint_pos": nav_obs[..., 6:18],
                "joint_vel": nav_obs[..., 18:30],
                "goal_commands": nav_obs[..., 30:34],  # 4 dims
                "forwards_depth_image": nav_obs[..., 34:290],
            }

        # Locomotion policy observations (Go1 version without height scan)
        if "low_level_policy" in self._env.obs_buf:
            locomotion_obs = self._env.obs_buf["low_level_policy"]
            obs_dict["locomotion_policy"] = {
                "base_lin_vel": locomotion_obs[..., 0:3],
                "base_ang_vel": locomotion_obs[..., 3:6],
                "projected_gravity": locomotion_obs[..., 6:9],
                "velocity_commands": locomotion_obs[..., 9:12],
                "joint_pos": locomotion_obs[..., 12:24],
                "joint_vel": locomotion_obs[..., 24:36],
                "actions": locomotion_obs[..., 36:48],
            }

        return "obs", obs_dict


@configclass
class PreStepNavigationAndLocomotionObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for PreStepNavigationAndLocomotionObservationsRecorder."""

    class_type: type[RecorderTerm] = PreStepNavigationAndLocomotionObservationsRecorder


# =============================================================================
# Recorder Manager Configurations for IL Data Collection
# =============================================================================


@configclass
class NavigationILRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager for navigation IL data collection.

    Records navigation policy observations in dict format (for robomimic),
    along with standard action and state recordings.

    Use this for Go1NavTasksDepthNavEnvCfg_IL_COLLECT where observations
    are concatenated (RL training mode) but recorded as dict for robomimic.
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    # Use custom dict-format recorder for navigation policy observations
    record_pre_step_policy_observations = PreStepNavigationPolicyObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()


@configclass
class NavigationILFullRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager that records both navigation and locomotion observations.

    Use this when you need full observability for analysis or when the IL policy
    requires both high-level navigation and low-level locomotion observations.
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    # Use custom dict-format recorder for both observation groups
    record_pre_step_all_observations = PreStepNavigationAndLocomotionObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()
