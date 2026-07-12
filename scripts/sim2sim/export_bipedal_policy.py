# Export the trained bipedal policy (TorchScript) + environment metadata for sim2sim.
# Produces <run_dir>/exported/policy.pt and sim2sim_meta.json (joint order, defaults,
# control params) that the MuJoCo rollout script consumes.
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

import importlib.metadata as metadata
import json, os, torch
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
    print(f"[EXPORT] checkpoint: {rp}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(handle_deprecated_rsl_rl_checkpoint(rp, ver))

    export_dir = os.path.join(os.path.dirname(rp), "exported")
    if hasattr(runner, "export_policy_to_jit"):
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    else:
        from isaaclab_rl.rsl_rl import export_policy_as_jit
        policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        normalizer = getattr(runner, "obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    print(f"[EXPORT] policy.pt -> {export_dir}")

    robot = env.unwrapped.scene["robot"]
    meta = {
        "checkpoint": rp,
        "joint_names": list(robot.joint_names),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "kp": 30.0,
        "kd": 0.8,
        "action_scale": 0.25,
        "decimation": 4,
        "physics_dt": 0.005,
        "obs_order": [
            "base_lin_vel(3)", "base_ang_vel(3)", "projected_gravity(3)",
            "velocity_commands(3)", "joint_pos_rel(12)", "joint_vel(12)",
            "last_action(12)", "com_cop_body(3)",
        ],
        "feet_order_for_cop": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        "cop_eps": [0.0, 0.0, 1e-6, 1e-6],
        "torque_limits": {"hip": 23.7, "thigh": 23.7, "calf": 45.43},
    }
    with open(os.path.join(export_dir, "sim2sim_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[EXPORT] sim2sim_meta.json written; joint order: {meta['joint_names']}")
    env.close()


main()
simulation_app.close()
