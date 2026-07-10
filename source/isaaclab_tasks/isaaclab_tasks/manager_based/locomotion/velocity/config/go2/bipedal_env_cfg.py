# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 task: bipedal (rear-legs) walking -- faithful reproduction of TumblerNet.

Reward terms, weights, commands, randomizations, and control parameters are ported
1:1 from the authors' released training configuration (Xiao et al., npj Robotics
2025; github.com/arclab-hku/bipedal_locomotion_for_quadrupedal_robots, files
``outputs/random_dog/Imi/test_estimator/{train_cfg_robot.py,robot.py}`` -- the exact
config their published checkpoints were trained with). No curriculum: the paper
trains directly to the vertical posture with the orientation + pendulum rewards.

Frame convention: the paper tracks commands in the raw STANDING body frame (linear
commands vs the negated body (y, z) velocities; yaw vs body-x angular velocity --
their ``_reward_tracking_lin_vel``/``_ang_vel`` overrides), ported verbatim. This is
deliberate, not a quirk: in quadruped stance those axes are lateral/VERTICAL, so
tracking reward is unearnable on four legs -- it only unlocks once the robot stands
up. That posture gating is the funnel that pulls the policy into the bipedal stance;
replacing it with gravity-aligned tracking lets a quadruped gait farm the tracking
reward and training collapses (verified empirically).

Documented deviations from the paper: (1) the policy observes ground-truth base
velocity and CoM-CoP vector in place of the concurrently-trained estimator network
(sim-only work; the estimator exists for real-robot deployment); (2) no
motor-strength/PD-gain randomization (no such event in this Isaac Lab version;
sim2real measure); (3) Go2 instead of Go1 (same scale, same default stance).
The paper's ``only_positive_rewards`` clip (without which early termination pays
better than balancing) IS reproduced, via :class:`BipedalManagerBasedRLEnv`.
"""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from .rough_env_cfg import UnitreeGo2RoughEnvCfg

# feet in a fixed order shared by every term below (asset and contact sensor agree)
FEET_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class BipedalManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with legged_gym's ``only_positive_rewards`` behavior.

    The paper clips the summed per-step reward at zero (their base config sets
    ``only_positive_rewards = True``). Without it, the heavy early penalties make
    terminating (falling) pay better than learning to balance.
    """

    def step(self, action):
        obs, rew, terminated, truncated, extras = super().step(action)
        rew.clamp_(min=0.0)
        return obs, rew, terminated, truncated, extras


##
# Rewards ported from the paper (gravity/world frame where the paper's standing body
# frame is used -- identical when upright, singularity-free during the transition).
##


def track_lin_vel_standing_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Paper's ``tracking_lin_vel``, verbatim: commands vs NEGATED body-frame (y, z) velocity.

    cmd_x pairs with -vel_body_y (lateral once standing) and cmd_y with -vel_body_z
    (forward = belly direction once standing). Unearnable in quadruped stance, where
    body z is vertical -- see the module docstring.
    """
    asset = env.scene[asset_cfg.name]
    err = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] + asset.data.root_lin_vel_b[:, 1:3]),
        dim=1,
    )
    return torch.exp(-err / std**2)


def track_ang_vel_standing_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Paper's ``tracking_ang_vel``, verbatim: yaw command vs body-x angular velocity.

    Body x is the world-up (yaw) axis once the robot is standing; in quadruped stance
    it is the roll axis, so yaw tracking too only unlocks when upright.
    """
    asset = env.scene[asset_cfg.name]
    err = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 0])
    return torch.exp(-err / std**2)


def gravity_xy_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Paper's ``orientation`` term (weight +0.8): g_x^2 + g_y^2 of projected gravity.

    Maximal (1.0) when gravity is perpendicular to the body z-axis, i.e. the trunk is
    vertical -- this POSITIVE reward is what pulls the robot up onto two legs.
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def gravity_z_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Paper's ``orientation_3`` term (weight -0.03): g_z^2, redundant push to vertical."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 2])


def front_feet_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Continuous penalty on front-feet contact force magnitude.

    The paper penalizes ||F_FL|| and ||F_FR|| (their fl/fr/f_contact_force terms,
    -0.03 each -> -0.06 per foot in total) instead of a binary contact flag.
    """
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).sum(dim=1)


class PendulumReward(ManagerTermBase):
    """TumblerNet's CoM-CoP stability terms (their r5; VHIP + cart-table models).

    Pendulum vector = CoM - CoP in the world frame, with the CoP force-weighted over
    all four feet (tiny bias on the rear feet keeps it defined in flight). This is
    the ``pen_vec`` the authors' SHIPPED env computes in its observation pipeline and
    uses in its reward overrides -- the same vector the policy observes as c-hat, so
    reward and observation agree exactly as in the paper. Modes:

    - ``"angle"``: theta^2, theta = angle of the pendulum vector from vertical
      (their ``inv_pendulum``, weight -0.1)
    - ``"acc"``: (sin(theta)/L)^2 ~ pendulum angular acceleration / g
      (their ``inv_pendulum_acc``, weight -0.0001)
    - ``"len_xy"``: ||pen_xy||, the cart-table handle length
      (their ``cart_table_len_xy``, weight -0.1)
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
    vector defined in flight, as in their code) to the mass-weighted CoM, expressed
    in the base frame.
    """

    def __init__(self, cfg: ObsTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene["robot"]
        self._sensor = env.scene.sensors["contact_forces"]
        self._feet_ids = self._asset.find_bodies(FEET_ORDER, preserve_order=True)[0]
        self._sensor_feet_ids = self._sensor.find_bodies(FEET_ORDER, preserve_order=True)[0]
        masses = self._asset.data.default_mass.to(env.device)
        self._mass = masses.unsqueeze(-1)
        self._total_mass = masses.sum(dim=1, keepdim=True)
        # paper: + [0, 0, 1e-6, 1e-6] so the rear feet anchor the CoP at zero contact
        self._eps = torch.tensor([0.0, 0.0, 1e-6, 1e-6], device=env.device)

    def __call__(self, env) -> torch.Tensor:
        feet_w = self._asset.data.body_pos_w[:, self._feet_ids, :]
        fz = self._sensor.data.net_forces_w[:, self._sensor_feet_ids, 2].clamp(min=0.0) + self._eps
        cop_w = (feet_w * fz.unsqueeze(-1)).sum(dim=1) / fz.sum(dim=1, keepdim=True)
        com_w = (self._mass * self._asset.data.body_com_pos_w).sum(dim=1) / self._total_mass
        return quat_apply_inverse(self._asset.data.root_quat_w, com_w - cop_w)


@configclass
class BipedalRewardsCfg(RewardsCfg):
    """TumblerNet's released reward set (their shipped train_cfg_robot.py scales)."""

    # -- r1 tracking: 1.0 / 0.6, sigma^2 = 0.25 (std 0.5), STANDING body frame (verbatim)
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_standing_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_standing_frame_exp,
        weight=0.6,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # -- r3 smoothness: body-frame velocity penalties, verbatim (their lin_vel_z -2.0
    # doubles as a speed cap once standing, since body z is then the forward axis)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # -- r4 bipedal encouragement: upright trunk + standing height
    orientation_up = RewTerm(func=gravity_xy_sq, weight=0.8)
    orientation_z = RewTerm(func=gravity_z_sq, weight=-0.03)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.55})
    # -- r4: front feet carry no load (continuous force, their fl+fr+f terms)
    front_feet_force = RewTerm(
        func=front_feet_force,
        weight=-0.06,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["F[LR]_foot"])},
    )
    # -- r5 stability: VHIP + cart-table on the CoM-CoP pendulum
    inv_pendulum = RewTerm(func=PendulumReward, weight=-0.1, params={"mode": "angle"})
    inv_pendulum_acc = RewTerm(func=PendulumReward, weight=-0.0001, params={"mode": "acc"})
    cart_table_len_xy = RewTerm(func=PendulumReward, weight=-0.1, params={"mode": "len_xy"})
    # -- r3 motion: joint deviation from the default stance (their *_motion terms;
    # rear hips strongest at -0.15, everything else -0.05)
    joint_deviation_f_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["F[LR]_hip_joint"])},
    )
    joint_deviation_r_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
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


@configclass
class BipedalCurriculumCfg:
    """No curriculum: the paper trains directly to the vertical posture."""

    pass


@configclass
class UnitreeGo2BipedalEnvCfg(UnitreeGo2RoughEnvCfg):
    rewards: BipedalRewardsCfg = BipedalRewardsCfg()
    curriculum: BipedalCurriculumCfg = BipedalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- terrain: flat plane (paper trains on flat ground only) ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # --- episode / control: paper values (22 s; action_scale 0.25, Kp 30, Kd 0.8) ---
        self.episode_length_s = 22.0
        self.actions.joint_pos.scale = 0.25
        self.scene.robot.actuators["base_legs"].stiffness = 30.0
        self.scene.robot.actuators["base_legs"].damping = 0.8

        # --- observations: the paper's actor also sees the CoM-CoP vector (c-hat) ---
        self.observations.policy.com_cop = ObsTerm(func=ComCopObs)

        # --- rewards: re-assert tracking weights (parent post_init sets quadruped
        # values 1.5/0.75 after our class-body defaults), then the remaining paper
        # scales on inherited terms ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.6
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_torques_l2.weight = 0.0  # paper: torques = -0.
        self.rewards.dof_acc_l2.weight = 0.0  # paper: dof_acc = -0.
        self.rewards.flat_orientation_l2.weight = 0.0  # replaced by orientation_up/_z
        # paper feet_air_time = 1.0, threshold 0.5 s, all feet (legged_gym first-contact style)
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
        # paper collision = -1.0 on thigh + calf contacts (their force threshold: 0.1 N)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 0.1,
            },
        )

        # --- commands: paper ranges (lin x/y and yaw all in [-1, 1], resample 10 s are
        # the base defaults). Direct yaw sampling instead of heading mode: Isaac Lab's
        # heading error uses the body-x forward axis, which points at the SKY once the
        # robot stands -- the paper's own heading mode used a bipedal forward axis. ---
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0

        # --- events: paper domain randomization ---
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
        # paper reset diversity (the quadruped parent config narrowed both)
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 10.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )

        # --- terminations: paper terminates on base/trunk/hip ground contact ---
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base", ".*_hip"]


@configclass
class UnitreeGo2BipedalEnvCfg_PLAY(UnitreeGo2BipedalEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # a few robots on the flat, deterministic
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5

        # scripted demo: walk forward at a steady bipedal pace for the whole episode.
        # NOTE the paper's command mapping: cmd_y tracks -body_z velocity = FORWARD
        # (belly direction) once standing; cmd_x is (negated) lateral.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (100.0, 100.0)

        # deterministic playback
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
