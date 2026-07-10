# Probe: is the robot genuinely bipedal (pitch ~85 deg, base high, front feet off the
# ground) or crouch-shuffling? Prints quantitative posture stats for one play robot.
import argparse, sys
from isaaclab.app import AppLauncher
import cli_args  # noqa
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Bipedal-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=600)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import math, os, torch
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
    print(f"[B] checkpoint: {rp}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(handle_deprecated_rsl_rl_checkpoint(rp, ver))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    contacts = env.unwrapped.scene["contact_forces"]
    front_ids = contacts.find_bodies(["F[LR]_foot"])[0]
    obs = env.get_observations()
    pitches, heights, front_hits = [], [], []
    for i in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs)); policy.reset(dones)
        g = robot.data.projected_gravity_b[0]
        pitch = math.degrees(math.atan2(-g[0].item(), -g[2].item()))
        z = robot.data.root_pos_w[0, 2].item()
        fmag = contacts.data.net_forces_w[0, front_ids].norm(dim=-1).max().item()
        pitches.append(pitch); heights.append(z); front_hits.append(1.0 if fmag > 1.0 else 0.0)
        if i % 50 == 0:
            print(f"[B] step {i:3d}  pitch={pitch:6.1f} deg  base_z={z:.3f}  front_contact={fmag > 1.0}")
    n = len(pitches) // 2  # stats over the second half (after standing up)
    second = pitches[n:]
    mp = sum(second) / len(second)
    mh = sum(heights[n:]) / len(second)
    fc = sum(front_hits[n:]) / len(second)
    print(f"[B] DONE (2nd half)  mean_pitch={mp:6.1f} deg  mean_base_z={mh:.3f}  front_contact_frac={fc:.2f}")
    print(f"[B] Interpretation: pitch >= ~75 deg, base_z >= ~0.45, front_contact_frac <= ~0.05 => genuinely bipedal.")
    env.close()


main()
simulation_app.close()
