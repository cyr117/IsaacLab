import argparse, sys
from isaaclab.app import AppLauncher
import cli_args  # noqa
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa
import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    origins = u.scene.env_origins
    print(f"[PROBE] env_origin (world) = {origins[0].tolist()}")
    print(f"[PROBE] robot spawn pos (world) = {robot.data.root_pos_w[0].tolist()}")
    # terrain sub-terrain origins (where each generated tile's spawn point is)
    ti = u.scene.terrain
    if hasattr(ti, "terrain_origins") and ti.terrain_origins is not None:
        print(f"[PROBE] terrain_origins = {ti.terrain_origins.reshape(-1, 3).tolist()}")
    # sample terrain height across x using the height scanner's ray hits at spawn
    hs = u.scene.sensors.get("height_scanner") if hasattr(u.scene, "sensors") else None
    if hs is not None:
        hits = hs.data.ray_hits_w[0]  # (num_rays, 3)
        xs = hits[:, 0]
        zs = hits[:, 2]
        print(f"[PROBE] height-scan x range under/around robot: {xs.min():.2f}..{xs.max():.2f}")
        print(f"[PROBE] height-scan z (terrain height) range: {zs.min():.2f}..{zs.max():.2f}  (flat if ~equal)")
    env.close()


main()
simulation_app.close()
