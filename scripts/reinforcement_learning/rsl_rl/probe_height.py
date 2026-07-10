# Probe: does the play robot actually CLIMB (base height rises) or circle on the flat?
import argparse, sys
from isaaclab.app import AppLauncher
import cli_args  # noqa
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=500)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os, torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
ver = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, ver)
    lp = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    rp = get_checkpoint_path(lp, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[H] checkpoint: {rp}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(handle_deprecated_rsl_rl_checkpoint(rp, ver))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    obs = env.get_observations()
    z0 = robot.data.root_pos_w[0, 2].item()
    zmax, zmin = z0, z0
    for i in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs)); policy.reset(dones)
        z = robot.data.root_pos_w[0, 2].item()
        zmax, zmin = max(zmax, z), min(zmin, z)
        if i % 50 == 0:
            print(f"[H] step {i:3d}  base_z={z:.3f}  (rise since spawn={z - z0:+.3f})")
    print(f"[H] DONE spawn_z={z0:.3f}  max_z={zmax:.3f}  min_z={zmin:.3f}  peak_rise={zmax - z0:+.3f}")
    print(f"[H] Interpretation: peak_rise > ~0.4 => climbed a pyramid; ~0 => stayed on flat (circled).")
    env.close()


main()
simulation_app.close()
