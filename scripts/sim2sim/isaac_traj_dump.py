# Record the policy's obs/action trajectory in Isaac from a DE-RANDOMIZED play reset
# (default joints, zero velocities, identity yaw) for step-by-step comparison against
# the MuJoCo sim2sim rollout started from the identical state.
import argparse, sys
from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/cyr/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Bipedal-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--out", type=str, default="/tmp/claude-1000/-home-cyr-legged-gym/5d83ca3d-94ac-4a61-a745-6ed272e2e34a/scratchpad/isaac_traj.npz")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import math, os, numpy as np, torch
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
    # de-randomize the reset completely
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    env_cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
    env_cfg.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
    env_cfg.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, ver)
    lp = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    rp = get_checkpoint_path(lp, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(handle_deprecated_rsl_rl_checkpoint(rp, ver))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    obs = env.get_observations()
    O, A, P = [], [], []
    for i in range(args_cli.steps):
        o = obs["policy"][0] if isinstance(obs, dict) else obs[0]
        with torch.inference_mode():
            a = policy(obs)
        O.append(o.detach().cpu().numpy().copy()); A.append(a[0].detach().cpu().numpy().copy())
        obs, _, dones, _ = env.step(a)
        policy.reset(dones)
        g = robot.data.projected_gravity_b[0]
        P.append(math.degrees(math.atan2(-g[0].item(), -g[2].item())))
        if i % 10 == 0:
            print(f"[TRAJ] step {i:3d} pitch={P[-1]:7.1f} z={robot.data.root_pos_w[0,2].item():.3f}")
    np.savez(args_cli.out, obs=np.array(O), act=np.array(A), pitch=np.array(P))
    print(f"[TRAJ] saved {args_cli.out}  checkpoint={rp}")
    env.close()


main()
simulation_app.close()
