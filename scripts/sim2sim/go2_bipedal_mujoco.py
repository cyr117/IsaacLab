# Sim2sim validation: run the Isaac Lab bipedal Go2 policy in MuJoCo.
#
# Replicates the training control stack (PD 30/0.8, action scale 0.25, 50 Hz policy
# over 5 ms physics) and assembles the same 51-dim observation from MuJoCo state,
# including the force-weighted CoM-CoP vector. Reports the same posture metrics as
# probe_bipedal.py so Isaac vs MuJoCo behavior is directly comparable.
#
# Usage (isaaclab conda env):
#   python scripts/sim2sim/go2_bipedal_mujoco.py --run 2026-07-12_14-18-35_scratch8
#   python scripts/sim2sim/go2_bipedal_mujoco.py --run ..._scratch8 --viewer   # watch live
import argparse
import glob
import json
import math
import os
import time

import mujoco
import numpy as np
import torch

MENAGERIE_XML = "/home/cyr/mujoco_menagerie/unitree_go2/scene.xml"
LOG_ROOT = "/home/cyr/IsaacLab/logs/rsl_rl/unitree_go2_bipedal"

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default=None, help="run dir name under logs (default: newest with exported/)")
parser.add_argument("--steps", type=int, default=1000, help="policy steps (50 Hz -> 20 s)")
parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0], help="velocity command [vx vy wz]")
parser.add_argument("--viewer", action="store_true", help="live viewer (real-time)")
parser.add_argument(
    "--init", type=str, default="quadruped", choices=["quadruped", "bipedal"],
    help="start pose: quadruped stance (tests stand-up too) or near-bipedal (tests balance/walk only)",
)
args = parser.parse_args()

# -- locate exported policy + metadata
if args.run:
    export_dir = os.path.join(LOG_ROOT, args.run, "exported")
else:
    candidates = sorted(glob.glob(os.path.join(LOG_ROOT, "*", "exported", "sim2sim_meta.json")))
    assert candidates, "no exported policy found; run export_bipedal_policy.py first"
    export_dir = os.path.dirname(candidates[-1])
meta = json.load(open(os.path.join(export_dir, "sim2sim_meta.json")))
policy = torch.jit.load(os.path.join(export_dir, "policy.pt")).eval()
print(f"[S2S] policy: {export_dir}  (from {meta['checkpoint']})")

# -- MuJoCo model
model = mujoco.MjModel.from_xml_path(MENAGERIE_XML)
model.opt.timestep = meta["physics_dt"]
data = mujoco.MjData(model)

# joint mapping: Isaac order (meta) -> MuJoCo qpos/qvel/actuator indices
joint_names = meta["joint_names"]
qadr, vadr, ctrl_adr, tau_lim = [], [], [], []
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"joint {name} not in MJCF"
    qadr.append(model.jnt_qposadr[jid])
    vadr.append(model.jnt_dofadr[jid])
    act = name.removesuffix("_joint")
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
    assert aid >= 0, f"actuator {act} not in MJCF"
    ctrl_adr.append(aid)
qadr, vadr, ctrl_adr = np.array(qadr), np.array(vadr), np.array(ctrl_adr)
default_q = np.array(meta["default_joint_pos"])

# Isaac trains Go2 with a DC-motor model on ALL joints (isaaclab_assets unitree.py):
# effort = saturation = 23.5 N*m, velocity limit 30 rad/s, torque fading with speed.
# Reproducing it here is essential -- flat URDF limits (45 N*m calves) over-drive
# the policy's maneuvers.
SAT_EFFORT, EFFORT_LIM, VEL_LIM = 23.5, 23.5, 30.0


def dc_motor_clip(tau, qd):
    max_tau = np.clip(SAT_EFFORT * (1.0 - qd / VEL_LIM), 0.0, EFFORT_LIM)
    min_tau = np.clip(SAT_EFFORT * (-1.0 - qd / VEL_LIM), -EFFORT_LIM, 0.0)
    return np.clip(tau, min_tau, max_tau)

# base body and foot geoms (foot = sphere geom in each calf body, Isaac CoP order)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
if base_id < 0:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
assert base_id >= 0, "base body not found"
foot_geoms = []
for leg in ["FL", "FR", "RL", "RR"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
    assert bid >= 0, f"{leg}_calf not found"
    gs = [g for g in range(model.ngeom)
          if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    # some menagerie variants attach the foot sphere to a dedicated foot body
    if not gs:
        fbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
        gs = [g for g in range(model.ngeom)
              if fbid >= 0 and model.geom_bodyid[g] == fbid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    assert gs, f"foot sphere geom for {leg} not found"
    foot_geoms.append(gs[0])
cop_eps = np.array(meta["cop_eps"])

# -- initial state
mujoco.mj_resetData(model, data)
if args.init == "quadruped":
    data.qpos[:3] = [0.0, 0.0, 0.34]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = default_q
else:  # near-bipedal (RSI-style): nose-up pitch, rear legs vertical under the body
    pitch0 = 1.35
    data.qpos[:3] = [0.0, 0.0, 0.45]
    data.qpos[3:7] = [math.cos(-pitch0 / 2), 0.0, math.sin(-pitch0 / 2), 0.0]  # w x y z, -pitch about y
    q0 = default_q.copy()
    for i, name in enumerate(joint_names):
        if name.startswith("R") and "thigh" in name:
            q0[i] = 1.0 + pitch0
        if name.startswith("R") and "calf" in name:
            q0[i] = -1.2
    data.qpos[qadr] = q0
mujoco.mj_forward(model, data)

kp, kd = meta["kp"], meta["kd"]
scale, decim = meta["action_scale"], meta["decimation"]
cmd = np.array(args.cmd)
heading_target = 0.0  # play config: hold world heading 0 (belly toward +x)
last_action = np.zeros(12)
vel6 = np.zeros(6)


def update_heading_cmd():
    """Training/play yaw command: 0.5 x wrap(heading_target - belly_heading), clip +-1."""
    R = data.xmat[base_id].reshape(3, 3)
    belly = R @ np.array([0.0, 0.0, -1.0])
    heading = math.atan2(belly[1], belly[0])
    err = (heading_target - heading + math.pi) % (2 * math.pi) - math.pi
    cmd[2] = float(np.clip(0.5 * err, -1.0, 1.0))


def foot_forces_z():
    fz = np.zeros(4)
    f6 = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        for k, g in enumerate(foot_geoms):
            if con.geom1 == g or con.geom2 == g:
                mujoco.mj_contactForce(model, data, i, f6)
                frame = con.frame.reshape(3, 3)  # rows = contact axes in world
                f_world = frame.T @ f6[:3]
                fz[k] += abs(f_world[2])
    return fz


def observe():
    R = data.xmat[base_id].reshape(3, 3)  # body -> world
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6, 1)  # local frame
    ang_b, lin_b = vel6[:3], vel6[3:]
    grav_b = R.T @ np.array([0.0, 0.0, -1.0])
    q = data.qpos[qadr] - default_q
    qd = data.qvel[vadr]
    # CoM-CoP in body frame (force-weighted CoP, rear-feet eps as in training)
    fz = np.clip(foot_forces_z(), 0.0, None) + cop_eps
    feet_pos = data.geom_xpos[foot_geoms]
    cop_w = (feet_pos * fz[:, None]).sum(axis=0) / fz.sum()
    com_w = data.subtree_com[base_id]
    com_cop_b = R.T @ (com_w - cop_w)
    return np.concatenate([lin_b, ang_b, grav_b, cmd, q, qd, last_action, com_cop_b]).astype(np.float32), fz


viewer = None
if args.viewer:
    import mujoco.viewer as mjviewer
    viewer = mjviewer.launch_passive(model, data)

pitches, heights, speeds, front_contact, fell_at = [], [], [], [], None
for step in range(args.steps):
    update_heading_cmd()
    obs, fz = observe()
    with torch.no_grad():
        last_action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    q_des = default_q + scale * last_action
    for _ in range(decim):
        qd = data.qvel[vadr]
        tau = kp * (q_des - data.qpos[qadr]) - kd * qd
        data.ctrl[ctrl_adr] = dc_motor_clip(tau, qd)
        mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()
        time.sleep(meta["physics_dt"] * decim)
    R = data.xmat[base_id].reshape(3, 3)
    g_b = R.T @ np.array([0.0, 0.0, -1.0])
    pitch = math.degrees(math.atan2(-g_b[0], -g_b[2]))
    z = data.qpos[2]
    speed = float(np.linalg.norm(data.qvel[:2]))
    pitches.append(pitch); heights.append(z); speeds.append(speed)
    front_contact.append(1.0 if (fz[0] > 1.0 or fz[1] > 1.0) else 0.0)
    if fell_at is None and step > 100 and z < 0.12:
        fell_at = step
    if step % 100 == 0:
        print(f"[S2S] step {step:4d}  pitch={pitch:6.1f} deg  z={z:.3f}  speed={speed:.2f}  fz={fz.round(1)}")

if viewer is not None:
    viewer.close()
n = len(pitches) // 2
print(
    f"[S2S] DONE (2nd half)  mean_pitch={sum(pitches[n:])/n:6.1f} deg  mean_z={sum(heights[n:])/n:.3f}"
    f"  front_contact_frac={sum(front_contact[n:])/n:.2f}  mean_speed_xy={sum(speeds[n:])/n:.2f}"
    f"  fell_at={'never' if fell_at is None else fell_at}"
)
print("[S2S] Isaac reference (same checkpoint, probe): compare pitch/z/front_contact/speed.")
