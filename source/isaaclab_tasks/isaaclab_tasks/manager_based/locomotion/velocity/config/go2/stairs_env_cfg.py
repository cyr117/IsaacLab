# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 task: approach a pyramid staircase from flat ground and traverse it UP then DOWN.

The robot spawns at the base (on flat ground, random yaw), is commanded to face +x and
walk forward, so it must: rotate to square up with the stairs -> climb up one face ->
cross the top platform -> walk down the far face. Training uses the same base-spawn
traverse so the policy actually learns the flat->stairs transition (the previous
center-spawn task never did, so it could only handle stairs it was already standing on).
"""

import numpy as np
import torch

import isaaclab.terrains as terrain_gen
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains.trimesh.mesh_terrains import pyramid_stairs_terrain
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


##
# Traverse terrain: same pyramid-stairs mesh, but spawn origin at the base (-x side,
# on the flat border, ground level) instead of the top platform. Walking +x then
# climbs UP one face, crosses the top platform, and descends the far face.
##
def pyramid_stairs_traverse_terrain(difficulty, cfg):
    meshes, _ = pyramid_stairs_terrain(difficulty, cfg)
    # spawn on the flat border just before the -x stairs (small run-up to square up)
    origin = np.array([0.5 * cfg.border_width, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class MeshPyramidStairsTraverseTerrainCfg(terrain_gen.MeshPyramidStairsTerrainCfg):
    function = pyramid_stairs_traverse_terrain


# training terrain: traverse pyramids across a difficulty curriculum (steps 0.05 -> 0.18 m)
STAIRS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,  # difficulty levels
    num_cols=20,  # variations
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stairs": MeshPyramidStairsTraverseTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.18),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# play terrain: a single traverse pyramid at fixed, comfortable step height
PLAY_TRAVERSE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 0.5),
    sub_terrains={
        "stairs": MeshPyramidStairsTraverseTerrainCfg(
            proportion=1.0,
            step_height_range=(0.10, 0.12),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


def base_roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize sideways (roll) tilt of the base only.

    Unlike ``flat_orientation_l2`` (roll AND pitch), this leaves pitch free — the body
    must pitch to match the stair slope — while pushing roll to zero. Climbing diagonally
    rolls the body, so this term rewards facing the stairs square-on.
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


class GaitReward(ManagerTermBase):
    """Trot-gait reward for quadrupeds (ported from Isaac Lab's Spot config).

    Rewards keeping the two diagonal foot pairs in sync with each other and out of sync
    with the opposing pair -> a trot in which all four legs cycle. This directly prevents
    the degenerate 3-legged gait where one leg is held in the air.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        pair_names = cfg.params["synced_feet_pair_names"]
        if len(pair_names) != 2 or len(pair_names[0]) != 2 or len(pair_names[1]) != 2:
            raise ValueError("This reward only supports two pairs of synchronized feet, like trotting.")
        pair_0 = self.contact_sensor.find_bodies(pair_names[0])[0]
        pair_1 = self.contact_sensor.find_bodies(pair_names[1])[0]
        self.synced_feet_pairs = [pair_0, pair_1]

    def __call__(self, env, std, max_err, velocity_threshold, synced_feet_pair_names, asset_cfg, sensor_cfg):
        sync_reward = self._sync(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1]) * self._sync(
            self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1]
        )
        async_reward = (
            self._async(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
            * self._async(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
            * self._async(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
            * self._async(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        )
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    def _sync(self, foot_0: int, foot_1: int) -> torch.Tensor:
        at = self.contact_sensor.data.current_air_time
        ct = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(torch.square(at[:, foot_0] - at[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(ct[:, foot_0] - ct[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async(self, foot_0: int, foot_1: int) -> torch.Tensor:
        at = self.contact_sensor.data.current_air_time
        ct = self.contact_sensor.data.current_contact_time
        se_0 = torch.clip(torch.square(at[:, foot_0] - ct[:, foot_1]), max=self.max_err**2)
        se_1 = torch.clip(torch.square(ct[:, foot_0] - at[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_0 + se_1) / self.std)


@configclass
class StairsRewardsCfg(RewardsCfg):
    """Default velocity-task rewards + stairs-specific terms."""

    base_roll_l2 = RewTerm(func=base_roll_l2, weight=-0.5)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    # enforce a trot so all four legs cycle (prevents the hanging-leg 3-legged gait)
    gait = RewTerm(
        func=GaitReward,
        weight=1.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


@configclass
class UnitreeGo2StairsEnvCfg(UnitreeGo2RoughEnvCfg):
    rewards: StairsRewardsCfg = StairsRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- terrain: traverse pyramids, start on the easiest and climb ---
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0

        # --- rewards ---
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = -1.0  # vertical motion is intrinsic to stairs
        # reward each foot for taking real steps (rough default 0.01 was so low the robot
        # dropped a leg to save torque); 0.25 is the shipped Go2-flat value
        self.rewards.feet_air_time.weight = 0.25
        # penalize walking on knees/shins: any contact on a thigh or calf link (feet are
        # the only parts that should touch). Re-enables the term Go2-rough disabled, with
        # Go2 link names. base_contact is already a termination, so it's not included here.
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )

        # --- command: rotate to face +x (into the stairs), then walk straight forward ---
        # spawn yaw is random (reset_base), so the robot must turn to square up; no
        # lateral command keeps it going straight up-over-down.
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.5)  # brisk: reward faster climbing
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class UnitreeGo2StairsEnvCfg_PLAY(UnitreeGo2StairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # one robot on a single traverse pyramid
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = PLAY_TRAVERSE_TERRAIN_CFG

        # scripted demo: face +x and keep walking forward for the whole episode
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.2)  # snappy watchable pace
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (100.0, 100.0)

        # deterministic playback
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
