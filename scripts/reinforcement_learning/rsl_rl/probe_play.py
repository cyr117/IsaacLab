# Diagnostic probe: load trained checkpoint, step policy headless, report if robots stay upright.
import argparse
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=300)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[PROBE] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    obs = env.get_observations()
    n_term = 0
    print(f"[PROBE] stepping {args_cli.steps} steps on {env.unwrapped.num_envs} envs...")
    for i in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
        n_term += int(dones.sum().item())
        if i % 50 == 0 or i == args_cli.steps - 1:
            gz = robot.data.projected_gravity_b[:, 2]          # -1 = perfectly upright, ~0 = tipped over
            speed = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
            print(f"[PROBE] step {i:3d} | upright(gravz) mean={gz.mean():.3f} min={gz.max():.3f} "
                  f"| base_speed mean={speed.mean():.3f} m/s | cumulative_terminations={n_term}")
    print(f"[PROBE] DONE. total_terminations={n_term} over {args_cli.steps} steps "
          f"({env.unwrapped.num_envs} envs). Interpretation: gravz~-1 & few terminations => standing/walking; "
          f"gravz~0 & many terminations => falling/collapsed.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
