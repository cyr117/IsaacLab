# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 task: stand up on the two REAR legs and walk bipedally on flat ground.

Approach (reproducing the behavior of TumblerNet, Xiao et al., npj Robotics 2025):
an upright-posture reward whose target pitch ramps from ~45 deg to ~85 deg via a
performance-gated global curriculum, a penalty on any non-rear-foot ground contact
(front feet, thighs, calves -- this also kills the sitting-dog local optimum), a
raised base-height target, a CoM-over-support stability penalty (stand-in for the
paper's VHIP/cart-table CoM-CoP rewards), and velocity tracking (forward/backward +
yaw) in the gravity-aligned frame -- body-frame tracking breaks once the body is
vertical. Not reproduced from the paper: the concurrent estimator network (sim-only
policy; ground-truth base velocity is observed directly) and lateral-velocity
commands (out of scope by design). Every episode starts in the normal quadruped
stance, so the policy also learns the stand-up maneuver itself.
"""

import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from .rough_env_cfg import UnitreeGo2RoughEnvCfg

##
# Curriculum target ranges. Level 0 is an easy rear-up; level 1 is near-vertical.
# Height: base sits ~0.34 m in quadruped stance (spawn 0.4 minus settling); a
# near-vertical stand on bent rear legs puts the base center around 0.55 m. The
# exact top value is verified with the Task 2 probe and tuned if needed.
##
PITCH_MIN, PITCH_MAX = math.radians(45.0), math.radians(85.0)
HEIGHT_MIN, HEIGHT_MAX = 0.34, 0.55


def _pitch_target(env: ManagerBasedRLEnv) -> float:
    # play/eval envs have no curriculum -> default to the final (vertical) target
    return getattr(env, "bipedal_pitch_target", PITCH_MAX)


def _height_target(env: ManagerBasedRLEnv) -> float:
    return getattr(env, "bipedal_height_target", HEIGHT_MAX)


def upright_pitch_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward matching the base attitude to the current curriculum pitch target.

    For a nose-up pitch theta (rotation about body y), gravity expressed in the base
    frame is (-sin(theta), 0, -cos(theta)). Matching the FULL projected-gravity vector
    drives pitch to the target and roll to zero in one term.
    """
    asset = env.scene[asset_cfg.name]
    theta = _pitch_target(env)
    g = asset.data.projected_gravity_b
    target = torch.tensor([-math.sin(theta), 0.0, -math.cos(theta)], device=g.device)
    err = torch.sum(torch.square(g - target), dim=1)
    return torch.exp(-err / std)


def base_height_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward holding the base at the curriculum standing height (stand tall, don't crouch)."""
    asset = env.scene[asset_cfg.name]
    err = torch.square(asset.data.root_pos_w[:, 2] - _height_target(env))
    return torch.exp(-err / std)


def lin_vel_z_world_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize WORLD-frame vertical velocity (bobbing).

    The default ``lin_vel_z_l2`` penalizes BODY-frame z velocity, which points
    backward-horizontal once the robot stands up -- it would punish forward walking.
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def ang_vel_xy_world_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize WORLD-frame roll/pitch rates (wobble), leaving commanded yaw free.

    The default ``ang_vel_xy_l2`` is BODY-frame: when the robot is vertical, commanded
    yaw turning appears as body-x angular velocity and would be punished.
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w[:, :2]), dim=1)


class ComOverSupportReward(ManagerTermBase):
    """Penalty: horizontal (world-xy) offset of the robot's CoM from the rear-feet midpoint.

    Stand-in for TumblerNet's CoM-CoP stability rewards (VHIP pendulum angle and
    cart-table handle length): standing is stable exactly when the CoM projects over
    the rear-feet support. Uses spawn-default link masses (startup mass randomization
    shifts them by at most a few percent).
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.params["asset_cfg"].name]
        self._feet_ids = self._asset.find_bodies(cfg.params["support_body_names"])[0]
        masses = self._asset.data.default_mass.to(env.device)  # (num_envs, num_bodies)
        self._mass = masses.unsqueeze(-1)
        self._total_mass = masses.sum(dim=1, keepdim=True)

    def __call__(self, env, asset_cfg, support_body_names):
        body_pos = self._asset.data.body_pos_w  # (num_envs, num_bodies, 3)
        com_xy = (self._mass * body_pos).sum(dim=1)[:, :2] / self._total_mass
        support_xy = body_pos[:, self._feet_ids, :2].mean(dim=1)
        return torch.sum(torch.square(com_xy - support_xy), dim=1)


class upright_curriculum(ManagerTermBase):
    """Global posture curriculum: ramps the pitch/height targets as the population succeeds.

    Tracks an EMA of the population's mean absolute pitch error. When the EMA drops
    below ``err_threshold`` the level steps up by ``level_step`` (0 -> 1 overall) and
    the EMA is bumped pessimistically so the next promotion waits for stable
    performance at the NEW target. Returns a dict of ``{"level": ..., "ema_err": ...}``
    so both appear in TensorBoard, as ``Curriculum/upright_level/level`` and
    ``Curriculum/upright_level/ema_err``.
    """

    def __init__(self, cfg: CurrTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.level = 0.0
        self.ema_err = 1.0  # pessimistic init (rad)
        self._apply(env)

    def _apply(self, env: ManagerBasedRLEnv):
        env.bipedal_pitch_target = PITCH_MIN + self.level * (PITCH_MAX - PITCH_MIN)
        env.bipedal_height_target = HEIGHT_MIN + self.level * (HEIGHT_MAX - HEIGHT_MIN)

    def __call__(self, env, env_ids, err_threshold: float = 0.15, level_step: float = 0.05, ema_alpha: float = 0.02):
        g = env.scene["robot"].data.projected_gravity_b
        pitch = torch.atan2(-g[:, 0], -g[:, 2])  # 0 = flat, pi/2 = vertical
        err = torch.abs(pitch - env.bipedal_pitch_target).mean().item()
        self.ema_err = (1.0 - ema_alpha) * self.ema_err + ema_alpha * err
        if self.ema_err < err_threshold and self.level < 1.0:
            self.level = min(1.0, self.level + level_step)
            self.ema_err += 0.2  # hysteresis: re-earn the threshold at the new target
            self._apply(env)
        return {"level": self.level, "ema_err": self.ema_err}


@configclass
class BipedalRewardsCfg(RewardsCfg):
    """Default velocity-task rewards + bipedal posture terms."""

    # -- task terms, frame-corrected for a pitched-up body (same pattern as the G1/H1
    # humanoid configs): the default BODY-frame trackers break once the robot is
    # vertical -- body x points at the sky, so tracking lin_vel_x would demand
    # vertical motion.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # world-frame regularizers (the body-frame defaults punish walking/turning when vertical)
    lin_vel_z_l2 = RewTerm(func=lin_vel_z_world_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=ang_vel_xy_world_l2, weight=-0.05)

    # the star reward: base attitude at the curriculum pitch target, roll ~ 0
    upright = RewTerm(func=upright_pitch_exp, weight=3.0, params={"std": 0.25})
    # stand tall at the curriculum height (prevents crouch-shuffling)
    base_height = RewTerm(func=base_height_exp, weight=1.0, params={"std": 0.05})
    # front feet must leave (and stay off) the ground
    front_feet_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["F[LR]_foot"]), "threshold": 1.0},
    )
    # keep legs under the body (no splaying / flailing sideways)
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    # front legs hang in a natural tucked pose (paper ablation: without a pose
    # regularizer the front legs cross and jitter)
    joint_deviation_front_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["F[LR]_thigh_joint", "F[LR]_calf_joint"])},
    )
    # keep the CoM over the rear-feet support (TumblerNet's core stability criterion)
    com_over_support = RewTerm(
        func=ComOverSupportReward,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "support_body_names": ["R[LR]_foot"]},
    )
    # crisp rear footholds (no skating)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names="R[LR]_foot"),
        },
    )


@configclass
class BipedalCurriculumCfg:
    """Posture curriculum only (flat ground -> no terrain curriculum)."""

    upright_level = CurrTerm(
        func=upright_curriculum,
        params={"err_threshold": 0.15, "level_step": 0.05, "ema_alpha": 0.02},
    )


@configclass
class UnitreeGo2BipedalEnvCfg(UnitreeGo2RoughEnvCfg):
    rewards: BipedalRewardsCfg = BipedalRewardsCfg()
    curriculum: BipedalCurriculumCfg = BipedalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- terrain: flat plane; balancing is the whole challenge ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # --- actions: bigger excursions than the quadruped tasks (rearing up needs
        # the rear hips/thighs far from the default stance) ---
        self.actions.joint_pos.scale = 0.5

        # --- rewards ---
        # re-assert the task weights: the parent (quadruped) post_init overrides them
        # after our class-body defaults are constructed
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.flat_orientation_l2.weight = 0.0  # replaced by the upright target
        # rear feet must actually step: biped air-time reward on the rear pair
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "threshold": 0.4,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R[LR]_foot"]),
            },
        )
        # nothing but the rear feet may rest on the ground (also kills the
        # sitting-dog pose: pitched-up body resting on the rear calves)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )

        # --- commands: forward/backward + turn, no lateral, some pure standing ---
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.rel_standing_envs = 0.2

        # --- terminations: falling flat ends the episode (base contact is inherited) ---
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.12}
        )


@configclass
class UnitreeGo2BipedalEnvCfg_PLAY(UnitreeGo2BipedalEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # a few robots on the flat, deterministic
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5

        # no curriculum at play time -> rewards fall back to the final vertical target
        self.curriculum.upright_level = None

        # scripted demo: walk forward at a steady bipedal pace for the whole episode
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (100.0, 100.0)

        # deterministic playback
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
