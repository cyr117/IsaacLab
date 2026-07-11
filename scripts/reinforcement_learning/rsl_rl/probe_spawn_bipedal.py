# Probe: why do bipedal-task episodes terminate at spawn? Zero-action rollout on the
# TRAIN env; reports per-body-group ground-contact fractions and base state over time.
import argparse, sys
from isaaclab.app import AppLauncher
import cli_args  # noqa
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Bipedal-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=150)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    sensor = u.scene.sensors["contact_forces"]
    names = sensor.body_names
    groups = {
        "base": [i for i, n in enumerate(names) if n == "base"],
        "hip": [i for i, n in enumerate(names) if n.endswith("_hip")],
        "thigh": [i for i, n in enumerate(names) if n.endswith("_thigh")],
        "calf": [i for i, n in enumerate(names) if n.endswith("_calf")],
        "foot": [i for i, n in enumerate(names) if n.endswith("_foot")],
    }
    robot = u.scene["robot"]
    zero = torch.zeros(args_cli.num_envs, 12, device=u.device)
    hits = {k: torch.zeros(args_cli.num_envs, device=u.device, dtype=torch.bool) for k in groups}
    died = torch.zeros(args_cli.num_envs, device=u.device, dtype=torch.bool)
    for step in range(args_cli.steps):
        obs, rew, term, trunc, info = env.step(zero)
        f = sensor.data.net_forces_w
        for k, ids in groups.items():
            hits[k] |= f[:, ids, :].norm(dim=-1).amax(dim=1) > 1.0
        died |= term
        if step in (5, 10, 25, 50, 100, args_cli.steps - 1):
            g = robot.data.projected_gravity_b
            pitch = (torch.atan2(-g[:, 0], -g[:, 2]).abs().mean() * 57.3).item()
            z = robot.data.root_pos_w[:, 2].mean().item()
            print(
                f"[PROBE] step {step:3d}: mean_z={z:.3f} mean_pitch_deg={pitch:5.1f} "
                f"died={died.float().mean().item():.2f} "
                + " ".join(f"{k}={hits[k].float().mean().item():.2f}" for k in groups),
                flush=True,
            )
    print("[PROBE] cumulative >1N ground-contact fraction (zero actions):")
    for k in groups:
        print(f"[PROBE]   {k}: {hits[k].float().mean().item():.3f}")
    env.close()


main()
simulation_app.close()
