# Go2 Bipedal Locomotion (walk on 2 rear legs) — Design

**Date:** 2026-07-10
**Status:** Approved (user selected: full velocity command, near-vertical ~80–90° posture, pitch-curriculum strategy)
**Reference:** Nature npj Robotics paper on bipedal quadruped walking (https://www.nature.com/articles/s44182-025-00043-2 — paywalled; design follows the standard approach that family of work shares: upright-posture reward + front-contact penalty + posture curriculum + velocity tracking on the rear legs).

## Goal

Train the Unitree Go2 to rear up onto its two hind legs and walk bipedally on flat
ground, tracking a full velocity command: forward/backward (`lin_vel_x`) and turning
(`ang_vel_z`). Lateral bipedal stepping (`lin_vel_y`) is explicitly out of scope.
Target posture is near-vertical (base pitch ~80–90° at the end of the curriculum).

## Non-goals

- Lateral (sideways) bipedal walking.
- Rough terrain / stairs while bipedal.
- Sim-to-real export (may follow later).

## Approach (chosen: single task with pitch curriculum)

One flat-terrain task trained from scratch. The upright-pitch target starts easy
(~45°) and ramps to ~85° via a performance-gated curriculum term, analogous to the
`terrain_levels` curriculum in the stairs task. Rejected alternatives:

- **Two-stage stand-then-walk** (train balance at 85° first, then fine-tune with
  velocity commands): more robust but two runs; kept as fallback if the curriculum
  stalls.
- **Fixed vertical target, no curriculum**: simplest config but a front-heavy Go2 is
  very unlikely to discover an ~85° balance from scratch; rejected.

## Files & registration (mirrors the stairs task)

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/bipedal_env_cfg.py`
  — new file: curriculum term, reward terms, `UnitreeGo2BipedalEnvCfg`,
  `UnitreeGo2BipedalEnvCfg_PLAY`. Subclasses `UnitreeGo2RoughEnvCfg`.
- `.../config/go2/agents/rsl_rl_ppo_cfg.py` — add `UnitreeGo2BipedalPPORunnerCfg`
  (experiment_name `unitree_go2_bipedal`).
- `.../config/go2/__init__.py` — register `Isaac-Velocity-Bipedal-Unitree-Go2-v0`
  and `Isaac-Velocity-Bipedal-Unitree-Go2-Play-v0`.

## Environment

- **Terrain:** flat plane (override rough terrain generator with a plane). Balancing
  is the challenge; terrain difficulty would compound failure modes.
- **Legs:** front = `FL_*`, `FR_*`; rear = `RL_*`, `RR_*`. Front legs are NOT
  hard-tucked — any front-body ground contact is penalized and the tucked pose
  emerges from learning.

## Rewards

New terms (weights are implementation-tuned starting points):

| Term | Kind | Purpose |
|---|---|---|
| `upright_pitch` | exp reward on projected gravity vs current curriculum pitch target | the star reward: drives the base toward the target lean |
| `base_height_target` | exp reward on base z vs curriculum-scaled standing height (final ~0.45–0.50 m; exact value verified from robot data at implementation) | stand tall, don't crouch |
| `front_contact` | penalty (`undesired_contacts`) on `F[LR]_(foot\|thigh\|calf)` | keep front legs off the ground |
| `track_lin_vel_x_exp`, `track_ang_vel_z_exp` | exp tracking | follow velocity commands (x-only linear tracking replaces the default xy term) |
| `feet_air_time` (rear feet only) | reward | make the rear legs actually step |
| regularizers | `lin_vel_z_l2` (softened), `ang_vel_xy_l2`, `action_rate_l2`, `dof_torques_l2`, `dof_acc_l2`, `feet_slide` (rear) | smooth stable motion |

Removed/disabled from the quadruped defaults: `GaitReward` (diagonal-pair trot logic
is meaningless for 2 legs), default `track_lin_vel_xy_exp` (replaced by x-only),
`flat_orientation_l2` (would fight the intentional pitch).

## Curriculum (the key mechanism)

Custom curriculum term `upright_level`:

- Maintains a per-run scalar level in `[0, 1]` mapping to a target pitch
  ~45° → ~85° (and scaling the height target).
- Performance-gated: level increases when the population's upright reward (or mean
  achieved pitch error) clears a threshold; may decrease on regression.
- Logged as `Curriculum/upright_level` in TensorBoard.
- The `upright_pitch` and `base_height_target` reward terms read the current target
  from this shared state.

## Commands

- `lin_vel_x`: approx (−0.4, 0.8) m/s (reduced vs quadruped speeds)
- `lin_vel_y`: (0, 0)
- `ang_vel_z`: approx (−0.8, 0.8) rad/s
- `rel_standing_envs` > 0 (e.g. 0.2) so balance-in-place is also learned
- No heading command (direct yaw-rate tracking)

## Terminations

- Keep base-contact termination (falling on the body ends the episode).
- Add fall termination: base height below a collapse threshold once the curriculum
  is past the crouch phase (implementation detail: threshold tied to curriculum
  level so early low-pitch phases aren't punished).

## Play config

- Flat plane, few envs (1–10), deterministic (no pushes, no obs noise).
- Scripted command schedule: walk forward, then turn — long resampling window.
- Review via headless video (established workflow) plus a quantitative probe
  (pitch + base height over time, reusing the `probe_height.py` pattern) to confirm
  genuine upright walking rather than crouch-shuffling.

## Training & verification

- From scratch: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py
  --task=Isaac-Velocity-Bipedal-Unitree-Go2-v0 --headless` (+ run name).
- Success criteria: `Curriculum/upright_level` reaches ~1.0; velocity tracking error
  low at high level; probe shows sustained base pitch ≥ ~75–80° and near-zero front
  contacts while tracking commands; video shows recognizable two-legged walking.
- Known risk: curriculum stalls below vertical → fallback to the two-stage strategy.
