# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 task: bipedal (rear-legs) walking -- reproduction of TumblerNet (Xiao et al.,
npj Robotics 2025).

Ported 1:1 from a VERIFIED working Isaac Gym reproduction
(``~/Downloads/legged_gym/ucl_go2_bipedal_upright_plane``: converged in 6000
iterations, mean reward 24.7, 96% episode survival, tracking 0.81/1.0). Every reward
term, weight, command setting, randomization, and the default stance below mirrors
that code (its ``bipedal_dog_config_baseline.py`` + ``robot.py``).

Frame convention once the robot stands nose-up (body x-axis skyward, belly = -body z
= travel direction):

* ``cmd_x`` tracks the belly-forward speed ``-v_z``  (their ``tracking_lin_vel``)
* ``cmd_y`` tracks the body-lateral speed ``+v_y``
* ``cmd_yaw`` tracks ``omega_x / 3`` (body-x = world-up when standing)
* the ``lin_vel_z`` penalty is on BODY-X velocity = world-vertical when upright --
  this is what suppresses hopping/jumping gaits
* the yaw command is heading-driven, with the heading measured from the BELLY
  direction projected on the ground (their ``forward_vec = (0, 0, -1)``)

In quadruped stance these tracking axes are unearnable (body z is then vertical), so
the tracking + orientation rewards form the funnel that pulls the policy upright --
no curriculum, no special initialization: episodes start from a quadruped crouch
(all thighs 0.9, calves -1.8, base at z=0.5) exactly as in the working code.

Documented deviations (sim-only / framework differences):

* the policy observes ground-truth base velocity and CoM-CoP vector instead of the
  concurrently-trained estimator network (the estimator exists for real-robot
  deployment; the working code's actor consumes the same quantities, estimated);
* the critic shares the policy observations instead of the 225-dim privileged vector;
* no actuation lag (2 sim steps) or per-joint motor offset (+-0.02 rad)
  randomization -- sim2real measures without an off-the-shelf Isaac Lab equivalent;
  motor-strength randomization (PD gains x U(0.9, 1.1)) IS ported.

The legged_gym ``only_positive_rewards`` clip (per-step total clamped at zero) is
reproduced via :class:`BipedalManagerBasedRLEnv`.
"""

import math
import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from .rough_env_cfg import UnitreeGo2RoughEnvCfg

# feet in a fixed order shared by every term below (asset and contact sensor agree)
FEET_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class BipedalManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with legged_gym's ``only_positive_rewards`` behavior.

    The reference code clips the summed per-step reward at zero. Without it, the
    heavy early penalties make terminating (falling) pay better than balancing.
    """

    def step(self, action):
        obs, rew, terminated, truncated, extras = super().step(action)
        rew.clamp_(min=0.0)
        return obs, rew, terminated, truncated, extras


##
# Commands: heading measured from the belly direction (their forward_vec = (0,0,-1))
##


class BellyHeadingVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand with the bipedal forward axis and legged_gym extras.

    Differences from the base class, both from the working reference code:

    * the heading used for the heading->yaw-rate controller is the world-frame
      direction of the BELLY axis (-body z), which is the travel direction once the
      robot stands (the base class uses body x, which points at the sky when upright);
    * linear commands with norm < 0.2 m/s are zeroed on resample
      (legged_gym's "set small commands to zero").
    """

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        small = torch.norm(self.vel_command_b[env_ids, :2], dim=1) < 0.2
        self.vel_command_b[env_ids[small], :2] = 0.0

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            quat = self.robot.data.root_quat_w[env_ids]
            belly = quat_apply(quat, torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(len(env_ids), 3))
            heading_w = torch.atan2(belly[:, 1], belly[:, 0])
            heading_error = wrap_to_pi(self.heading_target[env_ids] - heading_w)
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class BellyHeadingVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = BellyHeadingVelocityCommand


##
# Rewards (names in comments = the reference code's reward functions)
##


def track_lin_vel_bipedal_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference ``tracking_lin_vel``: cmd_x vs -v_body_z (belly-forward once upright),
    cmd_y vs +v_body_y (lateral). Unearnable in quadruped stance, where body z is
    vertical -- see the module docstring."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel = asset.data.root_lin_vel_b
    err = torch.square(cmd[:, 0] + vel[:, 2]) + torch.square(cmd[:, 1] - vel[:, 1])
    return torch.exp(-err / std**2)


def track_ang_vel_bipedal_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference ``tracking_ang_vel``: yaw command vs (1/3) * body-x angular velocity
    (body x is the world-up / yaw axis once standing)."""
    asset = env.scene[asset_cfg.name]
    err = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 0] / 3.0
    )
    return torch.exp(-err / std**2)


def lin_vel_body_x_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reference ``lin_vel_z`` (weight -2.0): squared BODY-X velocity.

    Body x is world-vertical once the robot stands nose-up, so this penalizes
    vertical bouncing -- the anti-hopping term. (In quadruped stance it is the
    forward axis, further discouraging quadruped locomotion.)"""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])


def ang_vel_body_yz_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reference ``ang_vel_xy`` (weight -0.05): squared body y,z angular velocities =
    pitch/roll wobble of the standing body (yaw = body x is tracked, not penalized)."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, 1:3]), dim=1)


def gravity_xy_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reference ``orientation`` (weight +0.8): g_x^2 + g_y^2 of projected gravity.

    Maximal (1.0) when gravity is perpendicular to the body z-axis, i.e. the trunk is
    vertical -- this POSITIVE reward is what pulls the robot up onto two legs."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def gravity_z_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reference ``orientation_3`` (weight -0.03): g_z^2, redundant push to vertical."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 2])


def front_feet_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reference ``fl/fr/f_contact_force`` (-0.03 each = -0.06 per front foot total):
    continuous penalty on front-feet contact force magnitude."""
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).sum(dim=1)


class PendulumReward(ManagerTermBase):
    """TumblerNet's CoM-CoP stability terms (VHIP + cart-table models).

    Pendulum vector = CoM - CoP in the world frame, with the CoP force-weighted over
    all four feet (tiny bias on the rear feet keeps it defined in flight), exactly as
    the reference env computes its ``pen_vec``. Modes:

    - ``"angle"``: theta^2, theta = angle of the pendulum vector from vertical
      (``inv_pendulum``, weight -0.1)
    - ``"acc"``: (sin(theta)/L)^2 ~ pendulum angular acceleration / g
      (``inv_pendulum_acc``, weight -0.0001)
    - ``"len_xy"``: ||pen_xy||, the cart-table handle length
      (``cart_table_len_xy``, weight -0.1)
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene["robot"]
        self._sensor = env.scene.sensors["contact_forces"]
        self._feet_ids = self._asset.find_bodies(FEET_ORDER, preserve_order=True)[0]
        self._sensor_feet_ids = self._sensor.find_bodies(FEET_ORDER, preserve_order=True)[0]
        masses = self._asset.data.default_mass.to(env.device)  # (num_envs, num_bodies)
        self._mass = masses.unsqueeze(-1)
        self._total_mass = masses.sum(dim=1, keepdim=True)
        self._eps = torch.tensor([0.0, 0.0, 1e-6, 1e-6], device=env.device)

    def __call__(self, env, mode: str) -> torch.Tensor:
        feet_w = self._asset.data.body_pos_w[:, self._feet_ids, :]
        fz = self._sensor.data.net_forces_w[:, self._sensor_feet_ids, 2].clamp(min=0.0) + self._eps
        cop = (feet_w * fz.unsqueeze(-1)).sum(dim=1) / fz.sum(dim=1, keepdim=True)
        com = (self._mass * self._asset.data.body_com_pos_w).sum(dim=1) / self._total_mass
        pen = com - cop
        length = pen.norm(dim=1).clamp(min=1e-6)
        cos_theta = (pen[:, 2] / length).clamp(-1.0, 1.0)
        if mode == "angle":
            return torch.square(torch.acos(cos_theta))
        if mode == "acc":
            return (1.0 - torch.square(cos_theta)) / torch.square(length)
        return pen[:, :2].norm(dim=1)  # "len_xy"


class ComCopObs(ManagerTermBase):
    """Ground-truth CoM-CoP connection vector in the base frame (the paper's c-hat).

    Stands in for the estimator network's output that TumblerNet feeds its actor:
    force-weighted CoP over the four feet (tiny bias on the rear feet keeps the
    vector defined in flight, as in the reference code) to the mass-weighted CoM,
    expressed in the base frame."""

    def __init__(self, cfg: ObsTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene["robot"]
        self._sensor = env.scene.sensors["contact_forces"]
        self._feet_ids = self._asset.find_bodies(FEET_ORDER, preserve_order=True)[0]
        self._sensor_feet_ids = self._sensor.find_bodies(FEET_ORDER, preserve_order=True)[0]
        masses = self._asset.data.default_mass.to(env.device)
        self._mass = masses.unsqueeze(-1)
        self._total_mass = masses.sum(dim=1, keepdim=True)
        self._eps = torch.tensor([0.0, 0.0, 1e-6, 1e-6], device=env.device)

    def __call__(self, env) -> torch.Tensor:
        feet_w = self._asset.data.body_pos_w[:, self._feet_ids, :]
        fz = self._sensor.data.net_forces_w[:, self._sensor_feet_ids, 2].clamp(min=0.0) + self._eps
        cop_w = (feet_w * fz.unsqueeze(-1)).sum(dim=1) / fz.sum(dim=1, keepdim=True)
        com_w = (self._mass * self._asset.data.body_com_pos_w).sum(dim=1) / self._total_mass
        return quat_apply_inverse(self._asset.data.root_quat_w, com_w - cop_w)


class BipedalGaitReward(ManagerTermBase):
    """Alternating-gait reward for the two rear feet (EXTENSION, not in the paper).

    The paper's reward set does not discriminate between hopping and walking: a
    two-footed forward hop earns tracking and air-time just like stepping. This term
    rewards the rear feet being OUT of phase -- one in stance while the other swings
    -- which produced human-like alternation in earlier runs of this task.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        feet = self.contact_sensor.find_bodies(cfg.params["foot_names"])[0]
        if len(feet) != 2:
            raise ValueError(f"Expected exactly two rear feet, got body ids {feet}.")
        self.foot_0, self.foot_1 = feet

    def __call__(self, env, std, max_err, velocity_threshold, foot_names, asset_cfg, sensor_cfg):
        at = self.contact_sensor.data.current_air_time
        ct = self.contact_sensor.data.current_contact_time
        se_0 = torch.clip(torch.square(at[:, self.foot_0] - ct[:, self.foot_1]), max=self.max_err**2)
        se_1 = torch.clip(torch.square(ct[:, self.foot_0] - at[:, self.foot_1]), max=self.max_err**2)
        reward = torch.exp(-(se_0 + se_1) / self.std)
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_w[:, :2], dim=1)
        return torch.where(torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), reward, 0.0)


def rear_feet_fore_aft_split(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="R[LR]_foot"),
) -> torch.Tensor:
    """Penalize a fore-aft scissor stance of the rear feet (EXTENSION, not in the paper).

    Without this, the policy cheats balance with a permanent lunge -- one rear foot
    far ahead of the body, one far behind -- which is statically stable fore-aft
    instead of the paper's feet-under-hips posture. Separation along the TRAVEL
    (belly) direction up to ``threshold`` (a normal walking stride) is free; only the
    sustained wide split is charged.
    """
    asset = env.scene[asset_cfg.name]
    feet_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    diff = feet_xy[:, 0] - feet_xy[:, 1]
    belly = quat_apply(
        asset.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    )
    heading = torch.atan2(belly[:, 1], belly[:, 0])
    dx = diff[:, 0] * torch.cos(heading) + diff[:, 1] * torch.sin(heading)
    return torch.square(torch.clamp(torch.abs(dx) - threshold, min=0.0))


def rear_feet_double_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize BOTH rear feet airborne at once (EXTENSION, not in the paper).

    A flight phase = hopping; walking keeps one foot planted. Together with the gait
    term this eliminated pronking in earlier runs of this task.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return (torch.sum(in_contact.int(), dim=1) == 0).float()


@configclass
class BipedalRewardsCfg(RewardsCfg):
    """The reference code's reward set, term for term (weights re-asserted in
    ``__post_init__`` where the quadruped parent config would override them)."""

    # -- tracking: 1.0 / 0.5, sigma^2 = 0.25 (std 0.5), bipedal frame (verbatim)
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_bipedal_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_bipedal_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # -- reference lin_vel_z / ang_vel_xy, remapped to the standing body axes
    lin_vel_z_l2 = RewTerm(func=lin_vel_body_x_sq, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=ang_vel_body_yz_sq, weight=-0.05)
    # -- bipedal encouragement: upright trunk + standing height (target 0.6)
    orientation_up = RewTerm(func=gravity_xy_sq, weight=0.8)
    orientation_z = RewTerm(func=gravity_z_sq, weight=-0.03)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.6})
    # -- front feet carry no load (continuous force, their fl+fr+f terms)
    front_feet_force = RewTerm(
        func=front_feet_force,
        weight=-0.06,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["F[LR]_foot"])},
    )
    # -- stability: VHIP + cart-table on the CoM-CoP pendulum
    inv_pendulum = RewTerm(func=PendulumReward, weight=-0.1, params={"mode": "angle"})
    inv_pendulum_acc = RewTerm(func=PendulumReward, weight=-0.0001, params={"mode": "acc"})
    cart_table_len_xy = RewTerm(func=PendulumReward, weight=-0.1, params={"mode": "len_xy"})
    # -- joint deviation from the default stance (their *_motion terms: BOTH hip
    # pairs -0.15, thighs/calves -0.05)
    joint_deviation_f_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["F[LR]_hip_joint"])},
    )
    # reference weight is -0.15, doubled here: in Isaac Lab the policy holds ~2x the
    # reference's rear-hip abduction (legs splayed outward for roll stability), and
    # the reference curve shows this term stops improving after ~iter 2000 on its own
    joint_deviation_r_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["R[LR]_hip_joint"])},
    )
    joint_deviation_thigh = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint"])},
    )
    joint_deviation_calf = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])},
    )
    # -- EXTENSIONS beyond the paper (user requirement: human-like alternating walk,
    # no hopping -- the paper's set does not discriminate between the two)
    gait = RewTerm(
        func=BipedalGaitReward,
        weight=1.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.3,
            "foot_names": ["R[LR]_foot"],
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    rear_double_air = RewTerm(
        func=rear_feet_double_air,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R[LR]_foot"])},
    )
    # no fencer's-lunge stance: rear feet stay near each other fore-aft, as in the paper
    rear_feet_split = RewTerm(
        func=rear_feet_fore_aft_split,
        weight=-10.0,
        params={"threshold": 0.15, "asset_cfg": SceneEntityCfg("robot", body_names="R[LR]_foot")},
    )


@configclass
class BipedalCurriculumCfg:
    """No curriculum: the reference code trains directly to the vertical posture."""

    pass


@configclass
class UnitreeGo2BipedalEnvCfg(UnitreeGo2RoughEnvCfg):
    rewards: BipedalRewardsCfg = BipedalRewardsCfg()
    curriculum: BipedalCurriculumCfg = BipedalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- terrain: flat plane ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # --- robot: PAPER AUTHORS' default stance (quadruped standing), not the
        # friend's deep crouch (thigh 0.9 / calf -1.8 / z 0.5). In Isaac Lab's Go2
        # USD the deep crouch settles on calves + thighs (spawn probe: thigh contact
        # 58%), which both kills episodes under thigh termination and makes the
        # sit/tripod postures comfortable local optima (friendport2 plateaued in a
        # front-leg-crutch tripod, ~60 N on front feet). The shallower paper stance
        # keeps thighs clear of the ground so thigh termination can do its job. ---
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.34)
        self.scene.robot.init_state.joint_pos = {
            ".*_hip_joint": 0.0,
            "F[LR]_thigh_joint": 0.8,
            "R[LR]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        }

        # --- episode / control: 22 s; action_scale 0.25, Kp 30, Kd 0.8 ---
        self.episode_length_s = 22.0
        self.actions.joint_pos.scale = 0.25
        self.scene.robot.actuators["base_legs"].stiffness = 30.0
        self.scene.robot.actuators["base_legs"].damping = 0.8

        # --- observations: the actor also sees the CoM-CoP vector (c-hat) ---
        self.observations.policy.com_cop = ObsTerm(func=ComCopObs)

        # --- rewards: re-assert what the quadruped parent post_init overrides,
        # zero out its extra shaping terms ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0  # replaced by orientation_up/_z
        # reference feet_air_time = 0.5, threshold 0.5 s, all feet
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
        # reference collision = -1.0 on thigh + calf contacts (force threshold 0.1 N)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 0.1,
            },
        )

        # --- commands: all ranges [-1, 1], heading mode with the belly forward axis,
        # stiffness 0.5, resample every 10 s (all as in the reference code) ---
        self.commands.base_velocity = BellyHeadingVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=BellyHeadingVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )

        # --- events: reference domain randomization ---
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            },
        )
        # motor strength: PD gains scaled by U(0.9, 1.1) per joint
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        # reset diversity: joints x U(0.5, 1.5), root velocities U(-0.5, 0.5)
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
        # push every 15 s with up to 1 m/s (reference push_interval_s = 15)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(15.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )

        # --- terminations: base/hip/thigh ground contact (friend's setting; the
        # thigh rule is the anti-sit/anti-kneel pressure -- viable here because the
        # paper stance above keeps thighs off the ground at spawn) ---
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base", ".*_hip", ".*_thigh"]


@configclass
class UnitreeGo2BipedalEnvCfg_PLAY(UnitreeGo2BipedalEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # a few robots on the flat, deterministic
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5

        # scripted demo: stand up, then walk belly-forward at 0.5 m/s while the
        # heading controller holds the initial world heading. NOTE: cmd_x is the
        # FORWARD (belly-direction) speed in this task's frame convention.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (100.0, 100.0)

        # deterministic playback; start quadruped so the demo shows the stand-up
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
