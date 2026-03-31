# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a pre-trained RL policy and compute success rate."""

import argparse
import os
import sys
import time
from collections import defaultdict

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a pre-trained RL policy on Go1 navigation task.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Go1NavTasksDepthNavEnvCfg_PLAY", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results (JSON).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import nav_tasks.configs  # noqa: F401


class EvaluationStats:
    """Tracks evaluation statistics across multiple parallel environments."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.episode_count = 0
        self.success_count = 0
        self.termination_reasons = defaultdict(int)
        self.episode_lengths = []
        self.episode_returns = []

        # Per-environment tracking
        self._episode_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self._episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def update(self, env, rewards: torch.Tensor, dones: torch.Tensor):
        """Update statistics based on environment step results.

        Args:
            env: The environment instance (for accessing termination manager).
            rewards: Reward tensor from env.step().
            dones: Done flags from env.step().
        """
        self._episode_steps += 1
        self._episode_returns += rewards

        # Check which environments are done
        done_envs = dones.nonzero(as_tuple=True)[0]

        if len(done_envs) > 0:
            for env_id in done_envs:
                self.episode_count += 1
                env_id_item = env_id.item()

                # Record episode length and return
                self.episode_lengths.append(self._episode_steps[env_id_item].item())
                self.episode_returns.append(self._episode_returns[env_id_item].item())

                # Check termination reasons
                term_manager = env.unwrapped.termination_manager

                # Get all available termination term names
                term_names = term_manager.active_terms

                # Check each termination term and build debug info
                term_values = {}
                for name in term_names:
                    try:
                        term_values[name] = term_manager.get_term(name)[env_id_item].item()
                    except:
                        term_values[name] = "N/A"

                # Debug: print all termination term values
                print(f"[DEBUG] Env {env_id_item} terminated. All terms: {term_values}")

                # Check specific terms for success/failure classification
                goal_reached = term_values.get("goal_reached", False)
                success = term_values.get("success", False)
                time_out = term_values.get("time_out", False)
                base_contact = term_values.get("base_contact", False)
                leg_contact = term_values.get("leg_contact", False)

                # Determine termination reason and success
                # Success = goal_reached OR success (for IL configs)
                if goal_reached or success:
                    self.termination_reasons["goal_reached"] += 1
                    self.success_count += 1
                elif time_out:
                    self.termination_reasons["time_out"] += 1
                elif base_contact:
                    self.termination_reasons["base_contact"] += 1
                elif leg_contact:
                    self.termination_reasons["leg_contact"] += 1
                else:
                    # Find which term caused termination
                    triggered_terms = [name for name, val in term_values.items() if val == True]
                    reason = triggered_terms[0] if triggered_terms else "unknown"
                    self.termination_reasons[f"other ({reason})"] += 1

                # Reset per-environment tracking
                self._episode_steps[env_id_item] = 0
                self._episode_returns[env_id_item] = 0.0

    def get_summary(self) -> dict:
        """Get a summary of evaluation statistics."""
        summary = {
            "total_episodes": self.episode_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.episode_count if self.episode_count > 0 else 0.0,
            "termination_reasons": dict(self.termination_reasons),
            "mean_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "std_episode_length": np.std(self.episode_lengths) if self.episode_lengths else 0.0,
            "mean_episode_return": np.mean(self.episode_returns) if self.episode_returns else 0.0,
            "std_episode_return": np.std(self.episode_returns) if self.episode_returns else 0.0,
        }
        return summary

    def print_summary(self):
        """Print evaluation summary to console."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Success Count: {summary['success_count']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print("\nTermination Reasons:")
        for reason, count in summary['termination_reasons'].items():
            percentage = (count / summary['total_episodes']) * 100 if summary['total_episodes'] > 0 else 0
            print(f"  {reason}: {count} ({percentage:.1f}%)")
        print("\nEpisode Statistics:")
        print(f"  Mean Length: {summary['mean_episode_length']:.2f} ± {summary['std_episode_length']:.2f}")
        print(f"  Mean Return: {summary['mean_episode_return']:.2f} ± {summary['std_episode_return']:.2f}")
        print("=" * 60 + "\n")


def main():
    """Run evaluation of a pre-trained RL policy."""
    # Check required arguments
    if args_cli.checkpoint is None:
        parser.error("--checkpoint is required for evaluation")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Parse agent configuration
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Set observations to concatenated mode for RSL-RL
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.concatenate_terms = True

    # Disable curriculum for evaluation
    env_cfg.curriculum = None

    # Set fixed parameters for evaluation
    if hasattr(env_cfg, 'events') and hasattr(env_cfg.events, 'reset_base'):
        env_cfg.events.reset_base.params["yaw_range"] = (0, 0)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Create runner with full agent config
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Initialize evaluation statistics
    num_envs = env.unwrapped.num_envs
    stats = EvaluationStats(num_envs, device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    print(f"[INFO] Starting evaluation for {args_cli.num_episodes} episodes with {num_envs} parallel environments...")

    # simulate environment
    while simulation_app.is_running() and stats.episode_count < args_cli.num_episodes:
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)

            # Update statistics
            stats.update(env, rewards, dones)

            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Print and save results
    stats.print_summary()

    if args_cli.output:
        summary = stats.get_summary()
        summary["checkpoint"] = args_cli.checkpoint
        summary["task"] = args_cli.task
        summary["num_envs"] = num_envs
        summary["seed"] = args_cli.seed

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)

        with open(args_cli.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Results saved to: {args_cli.output}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
