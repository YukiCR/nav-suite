from __future__ import annotations

import os

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import DatasetExportMode
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, MeshRepeatedBoxesTerrainCfg, FlatPatchSamplingCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

from nav_suite.collectors import TrajectorySamplingCfg
from nav_suite.terrain_analysis import TerrainAnalysisCfg

import nav_tasks.mdp as mdp
from nav_tasks.mdp.recorders import NavigationILRecorderManagerCfg
from nav_tasks.sensors import ZED_X_MINI_WIDE_RAYCASTER_CFG, adjust_ray_caster_camera_image_size

from nav_tasks.configs.env_cfg_base import *
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG # base config of go1 robot

from datetime import datetime

DIS_OBS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=3.0,
    num_rows=15,
    num_cols=15,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "repeated_boxes": MeshRepeatedBoxesTerrainCfg(
                            proportion= 1.0,
                            object_params_start= MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                                                        num_objects=30,
                                                        height=2.0,
                                                        size=(0.5, 0.5),
                                                        max_yx_angle=0.0
                                                    ),
                            object_params_end= MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                                                        num_objects=60,
                                                        height=2.0,
                                                        size=(0.5, 0.5),
                                                        max_yx_angle=0.0
                                                    ),
                            platform_width=2.0,
                            platform_height=0.0,
                            rel_height_noise=(1.0, 1.0),
                ),

        }
)

DIS_OBS_STATIC_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=0.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    curriculum=False,
    use_cache=False,
    sub_terrains={
        "repeated_boxes": MeshRepeatedBoxesTerrainCfg(
                            proportion= 1.0,
                            object_params_start= MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                                                        num_objects=150,
                                                        height=2.0,
                                                        size=(0.5, 0.5),
                                                        max_yx_angle=0.0
                                                    ),
                            object_params_end= MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                                                        num_objects=150,
                                                        height=2.0,
                                                        size=(0.5, 0.5),
                                                        max_yx_angle=0.0
                                                    ),
                            platform_width=1.0,
                            platform_height=0.0,
                            rel_height_noise=(1.0, 1.0)
                ),

        }
)


@configclass
class Go1NavTasksDepthNavSceneCfg(NavTasksDepthNavSceneCfg):
    # Post initialization
    def __post_init__(self) -> None:
        super().__post_init__()

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=DIS_OBS_TERRAINS_CFG,
            max_init_terrain_level=None,
            collision_group=-1,
            # physics_material=sim_utils.RigidBodyMaterialCfg(
            #     friction_combine_mode="multiply",
            #     restitution_combine_mode="multiply",
            #     static_friction=1.0,
            #     dynamic_friction=1.0,
            # ),
            debug_vis=False,
        )

        self.robot : ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # SENSORS: Locomotion Policy
        # self.height_scanner = RayCasterCfg(
        #                 prim_path="{ENV_REGEX_NS}/Robot/trunk",
        #                 offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        #                 ray_alignment="yaw",
        #                 pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        #                 debug_vis=False,
        #                 mesh_prim_paths=["/World/ground"],
        #             )

        self.height_scanner = None # not needed when use flat locomotion policy

        # SENSORS: Navigation Policy
        self.front_zed_camera = ZED_X_MINI_WIDE_RAYCASTER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/trunk",
            mesh_prim_paths=TERRAIN_MESH_PATH,
            update_period=0,
            debug_vis=False,
            offset=RayCasterCameraCfg.OffsetCfg(
                # The camera can be mounted at either 10 or 15 degrees on the robot.
                # pos=(0.4761, 0.0035, 0.1055), rot=(0.9961947, 0.0, 0.087155, 0.0), convention="world"  # 10 degrees
                pos=(0.4761, 0.0035, 0.1055),
                rot=(0.9914449, 0.0, 0.1305262, 0.0),
                convention="world",  # 15 degrees
            ),
        )

        # down sample
        self.front_zed_camera = adjust_ray_caster_camera_image_size(
            self.front_zed_camera, IMAGE_SIZE_DOWNSAMPLE_FACTOR, IMAGE_SIZE_DOWNSAMPLE_FACTOR
        )

        # turn off the self-collisions
        self.robot.spawn.articulation_props.enabled_self_collisions = False


@configclass
class Go1ActionsCfg(ActionsCfg): 
    def __post_init__(self) -> None:

        self.velocity_command = mdp.NavigationSE2ActionCfg(
            asset_name="robot",
            low_level_decimation = 4,
            low_level_action=mdp.JointPositionActionCfg(
                # go1 needs 0.25 scale
                asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
            ),
            low_level_policy_file="exts/nav_tasks/data/Policies/go1_flat_locomotion_policy.pt",
        )

@configclass
class Go1ObservationsCfg(ObservationsCfg):
    def __post_init__(self) -> None:

        @configclass
        class Go1LocomotionPolicyCfg(ObsGroup):
            """Observations for policy group."""

            # observation terms (order preserved)
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            projected_gravity = ObsTerm(func=mdp.projected_gravity)
            velocity_commands = ObsTerm(func=mdp.vel_commands, params={"action_term": "velocity_command"})
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_low_level_action, params={"action_term": "velocity_command"})
            # no height scan

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.low_level_policy = Go1LocomotionPolicyCfg()
        # navigation policy observation unchanged


@configclass
class Go1TerminationsCfg(TerminationsCfg):
    def __post_init__(self) -> None:

        self.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk"]),
                "threshold": 0.0,
            },
            time_out=False,
        )

        self.leg_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*hip", ".*calf"]),
                "threshold": 0.0,
            },
            time_out=False,
        )

@configclass
class Go1NavTasksDepthNavEnvCfg(NavTasksDepthNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene = Go1NavTasksDepthNavSceneCfg(num_envs=64, env_spacing=8)
        self.observations = Go1ObservationsCfg()
        self.actions = Go1ActionsCfg()
        self.terminations = Go1TerminationsCfg()

        self.sim.render_interval = 20

        self.commands.goal_command = mdp.GoalCommandCfg(
                                        asset_name="robot",
                                        z_offset_spawn=0.0,
                                        num_pairs=1000,
                                        path_length_range=[5, 20],
                                        traj_sampling=TrajectorySamplingCfg(
                                            enable_saved_paths_loading=False,
                                            terrain_analysis=TerrainAnalysisCfg(
                                                raycaster_sensor="front_zed_camera",
                                                sample_points = 10000,
                                                max_terrain_size=100.0,
                                                max_path_length=20.0,
                                                semantic_cost_mapping=None,
                                                viz_graph=True,
                                                viz_height_map=False,
                                            ),
                                        ),
                                        resampling_time_range=(1.0e9, 1.0e9),  # No resampling
                                        debug_vis=True,
                                    )

        self.rewards.episode_termination.weight = -400 # more penalization on collision

        self.curriculum.goal_distances = CurrTerm(
                    func=mdp.modify_goal_distance_in_steps,
                    params={
                        "update_rate_steps": 50 * 48,
                        "min_path_length_range": (0.0, 2.0),
                        "max_path_length_range": (5.0, 20.0),
                        "step_range": (50 * 48, 2000 * 48),
                    },
                )

        # Gradually increase episode termination penalty
        # FIXME: the robot stops moving forward if the penalty is high, figure out the suitable range and schedule for the penalty
        self.curriculum.episode_termination_ramp = CurrTerm(
            func=mdp.change_reward_weight,
            params={
                "reward_name": "episode_termination",
                "weight_range": (-400.0, -500.0),
                "step_range": (500 * 48, 20000 * 48),
                "mode": "linear",
            },
        )

        # NOTE: not sure if this modification is correct
        self.terminations.time_out = DoneTerm(
                            func=mdp.proportional_time_out,
                            params={
                                "max_speed": 1.0,
                                "safety_factor": 6.0,
                            },
                            time_out=True,  # No termination penalty for time_out = True
                        )
        
        # update sensor update periods
        # We tick contact sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # We tick the cameras based on the navigation policy update period.
        if self.scene.front_zed_camera is not None:
            self.scene.front_zed_camera.update_period = self.decimation * self.sim.dt


@configclass 
class Go1NavTasksDepthNavEnvCfg_TRAIN(Go1NavTasksDepthNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()

@configclass
class Go1NavTasksDepthNavEnvCfg_PLAY(Go1NavTasksDepthNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Change number of environments
        self.scene.num_envs = 10

        # Disable curriculum
        self.curriculum = None

        # Set fixed parameters for play mode
        self.events.reset_base.params["yaw_range"] = (0, 0)
        self.terminations.goal_reached.params = {
            "time_threshold": 0.1,
            "distance_threshold": 0.5,
            "angle_threshold": 0.3,
            "speed_threshold": 0.6,
        }

        self.viewer.eye = (0.0, 7.0, 7.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.commands.goal_command = mdp.GoalCommandCfg(
                                        asset_name="robot",
                                        z_offset_spawn=0.0,
                                        num_pairs=1000,
                                        path_length_range=[5, 20],
                                        traj_sampling=TrajectorySamplingCfg(
                                            enable_saved_paths_loading=False,
                                            terrain_analysis=TerrainAnalysisCfg(
                                                raycaster_sensor="front_zed_camera",
                                                sample_points = 10000,
                                                max_terrain_size=100.0,
                                                semantic_cost_mapping=None,
                                                viz_graph=True,
                                                viz_height_map=False,
                                            ),
                                        ),
                                        resampling_time_range=(1.0e9, 1.0e9),  # No resampling
                                        debug_vis=True,
                                    )

        self.scene.num_envs = 8


@configclass
class Go1NavTasksDepthNavEnvCfg_DEV(Go1NavTasksDepthNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2

# =============================================================================
# IL (Imitation Learning) Observation Configurations
# =============================================================================

@configclass
class Go1NavTasksDepthNavEnvCfg_IL_COLLECT(Go1NavTasksDepthNavEnvCfg):
    """Environment config for collecting demonstrations with trained RL policy.

    Uses concatenated observations (same as training) but records them as dict
    via the recorder for robomimic compatibility.
    """
    def __post_init__(self):
        super().__post_init__()

        # change to a fixed map
        self.scene.terrain.terrain_generator = DIS_OBS_STATIC_TERRAINS_CFG
        # fixed seed for deterministic map generation
        self.seed = 42

        # Disable curriculum, else the path length will be impacted by the curriculum
        self.curriculum = None

        self.commands.goal_command = mdp.GoalCommandCfg(
                                        asset_name="robot",
                                        z_offset_spawn=0.0,
                                        num_pairs=1000,
                                        path_length_range=[5, 8],
                                        traj_sampling=TrajectorySamplingCfg(
                                            enable_saved_paths_loading=False,
                                            terrain_analysis=TerrainAnalysisCfg(
                                                raycaster_sensor="front_zed_camera",
                                                sample_points = 2000,
                                                max_terrain_size= 50.0,
                                                max_path_length= 20.0,
                                                semantic_cost_mapping=None,
                                                viz_graph=True,
                                                viz_height_map=False,
                                            ),
                                        ),
                                        resampling_time_range=(5, 10),  # Do resampling
                                        debug_vis=True,
                                    )
        
        # success key is needed for export successful episodes
        self.terminations.success = self.terminations.goal_reached

        # Configure recorder for IL data collection
        # Records observations as dict for robomimic compatibility
        self.recorders = NavigationILRecorderManagerCfg(
            dataset_export_dir_path="./datasets",
            dataset_filename="go1_nav_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
        )

@configclass
class Go1NavTasksDepthNavEnvCfg_IL_COLLECT_DEV(Go1NavTasksDepthNavEnvCfg_IL_COLLECT):
    """
    Go1NavTasksDepthNavEnvCfg_IL_COLLECT_DEV, only change num env to 1 for bugging
    """
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1


@configclass
class Go1ILObservationsCfg(Go1ObservationsCfg):
    """Observation config for IL training with dict-style (non-concatenated) observations."""

    def __post_init__(self):
        # do post init of base class first
        super().__post_init__()

        # Replace observation groups with non-concatenated versions
        self.policy.concatenate_terms = False # do not concatenate when using IL policy


@configclass
class Go1NavTasksDepthNavEnvCfg_IL_PLAY(Go1NavTasksDepthNavEnvCfg_IL_COLLECT):
    """Environment config for playing/validating IL policy.

    Uses non-concatenated (dict-style) observations which are required by
    robomimic policies. This allows the trained IL policy to receive observations
    in the same format as during training.
    """
    def __post_init__(self):
        super().__post_init__()

        # Use dict-style observations for robomimic compatibility
        self.observations = Go1ILObservationsCfg()

        # Use at_goal function (not StayedAtGoal class) for compatibility with play.py
        self.terminations.success = DoneTerm(
            func=mdp.at_goal,
            params={
                "distance_threshold": 0.5,
                "angle_threshold": 0.3,
                "speed_threshold": 0.6,
            },
            time_out=False,
        )

        # self.commands.goal_command = mdp.GoalCommandCfg(
        #                                 asset_name="robot",
        #                                 z_offset_spawn=0.0,
        #                                 num_pairs=1000,
        #                                 path_length_range=[5, 10],
        #                                 traj_sampling=TrajectorySamplingCfg(
        #                                     enable_saved_paths_loading=False,
        #                                     terrain_analysis=TerrainAnalysisCfg(
        #                                         raycaster_sensor="front_zed_camera",
        #                                         sample_points = 3000,
        #                                         max_terrain_size= 100.0,
        #                                         max_path_length=30.0,
        #                                         semantic_cost_mapping=None,
        #                                         viz_graph=False,
        #                                         viz_height_map=False,
        #                                     ),
        #                                 ),
        #                                 resampling_time_range=(5, 10),  # Do resampling
        #                                 debug_vis=True,
        #                             )

        # disable recorder
        self.recorders = None

        self.viewer.eye = (0.0, 0.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.scene.num_envs = 1