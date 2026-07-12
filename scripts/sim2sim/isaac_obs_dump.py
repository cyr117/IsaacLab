# Dump the 51-dim policy observation at a CLEAN quadruped spawn state (teleported:
# base at (0,0,0.34), identity quat, default joints, zero velocities) for A/B
# comparison against the MuJoCo sim2sim pipeline at the identical state.
import argparse, sys
from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/cyr/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Bipedal-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    # teleport to the clean reference state
    ids = torch.tensor([0], device=u.device)
    pos = u.scene.env_origins[[0]].clone()
    pos[:, 2] = 0.34
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=u.device)
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=u.device), env_ids=ids)
    q = robot.data.default_joint_pos[[0]].clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    u.sim.step(render=False)
    robot.update(u.physics_dt)
    u.scene.sensors["contact_forces"].update(u.physics_dt)
    obs = u.observation_manager.compute()["policy"][0]
    print("[ISAAC-OBS] cmd (from manager):", u.command_manager.get_command("base_velocity")[0].cpu().tolist())
    labels = ["lin_vel"] * 3 + ["ang_vel"] * 3 + ["grav"] * 3 + ["cmd"] * 3 + ["q"] * 12 + ["qd"] * 12 + ["act"] * 12 + ["comcop"] * 3
    for i in range(0, 51, 3):
        print(f"[ISAAC-OBS] {i:2d} {labels[i]:8s} {obs[i]:8.4f} {obs[i+1]:8.4f} {obs[i+2]:8.4f}")
    env.close()


main()
simulation_app.close()
