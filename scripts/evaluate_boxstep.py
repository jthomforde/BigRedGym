# AI-assisted: rollout logger and the four box-step metrics from BOXSTEP_PLAN.md
# ("Metrics"), plus reference-tracking diagnostics for the policy-off run.
"""Run the box step, log every control step to CSV, and compute the metrics.

Usage:
    uv run --frozen scripts/evaluate_boxstep.py --headless              # latest policy
    uv run --frozen scripts/evaluate_boxstep.py --headless --load_run <run> \
                                                --checkpoint 500
    uv run --frozen scripts/evaluate_boxstep.py --headless --reference_only

--reference_only runs the reference with zero policy corrections (the same thing
as the runner's actor.disable_actions = True) and loads no checkpoint.

Every robot starts its clock on beat 1. A robot's data is used only up to its
first termination (base contact). Outputs, under logs/:
    boxstep_evaluation_steps.csv     one row per control step per robot
    boxstep_evaluation_metrics.json  the metrics printed at the end

Definitions (thresholds are tunable constants below):
    stable   all four feet in contact, body tilt < 3 deg, |roll rate| and
             |pitch rate| < 0.2 rad/s, held for 100 ms.
    1. time to stability: on each hold beat, time from the start of the hold
       beat to the start of the first 100 ms stable window inside that beat.
    2. foot repositioning count: touchdowns per pair per beat beyond the ones
       scheduled (one per foot on a moving beat, none on a hold beat), summed
       over the pair's two feet.
    3. foot repositioning time: per foot and beat, air time after the foot's
       first touchdown in the beat (on hold beats, after the beat start if the
       foot starts down), averaged per pair.
    4. step timing: first touchdown of each foot on each moving beat minus its
       scheduled touchdown (lead pair at sigma*T_b, follow pair at 2*sigma*T_b).
"""

import argparse
import csv
import json
import math
import os
import random

import numpy as np
import torch

from gym.envs.go2 import boxstep_reference as reference
from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry

STEP_LOG_PATH = "logs/boxstep_evaluation_steps.csv"
METRICS_PATH = "logs/boxstep_evaluation_metrics.json"
STABLE_TILT_DEG = 3.0
STABLE_BODY_RATE_RAD_S = 0.2
STABLE_HOLD_S = 0.1
LEG_NAMES = ["FL", "FR", "RL", "RR"]
JOINT_NAMES = ["hip", "thigh", "calf"]
PAIR_NAMES = ["left", "right"]


# ------------------------------------------------------
#
# Setup
#
# ------------------------------------------------------


def get_evaluation_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the Go2 box step")
    parser.add_argument("--task", type=str, default="go2boxstep")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_boxes", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--reference_only",
        action="store_true",
        help="Zero policy corrections (disable_actions) and load no checkpoint.",
    )
    return parser.parse_args(argv)


def build_env_and_runner(args):
    """Build the env like scripts/play.py; the runner is None if reference-only."""
    import gym.envs  # noqa: F401  (registers the tasks)

    env_cfg, train_cfg = task_registry.get_cfgs(
        args.task,
        original_cfg=args.original_cfg,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
    )
    slowest_beat_s = 60.0 / min(env_cfg.boxstep.tempo_bpm_range)
    rollout_s = args.num_boxes * reference.BEATS_PER_BOX * slowest_beat_s
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = rollout_s + 1.0
    env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.seed = args.seed if args.seed is not None else random.randint(0, 10000)
    train_cfg.seed = env_cfg.seed
    train_cfg.runner.device = args.device
    train_cfg.logging.enable_local_saving = False
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(env_cfg.seed)
    env = task_registry.make_env(
        args.task, env_cfg, device=args.device, headless=args.headless
    )
    if args.reference_only:
        return env, None, rollout_s

    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    task_registry.set_log_dir_name(train_cfg)
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner, rollout_s


def restart_box_clock(env):
    """Start every robot's clock on beat 1 (the operator's key press)."""
    env.beat_time.zero_()
    env.phase.zero_()
    env._update_gait_reference()
    env._update_phase_observation()


# ------------------------------------------------------
#
# Rollout and logging
#
# ------------------------------------------------------


def compute_euler_angles(quaternion_xyzw):
    """Return roll, pitch, yaw (N,) from quaternions (N, 4) in x, y, z, w order."""
    x, y, z, w = quaternion_xyzw.unbind(dim=1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def record_step(env, applied_reference):
    """Return a dict of (N, ...) numpy arrays describing the robots right now."""
    beat_period = env.beat_period.squeeze(1)
    beat_time = env.beat_time.squeeze(1)
    beat_index, beat_fraction = reference.compute_beat_state(beat_time, beat_period)
    roll, pitch, yaw = compute_euler_angles(env.root_states[:, 3:7])
    tilt = torch.arccos(torch.clamp(-env.projected_gravity[:, 2], -1.0, 1.0))
    joint_pos = env.dof_pos.index_select(1, env.actuated_dof_indices)
    record = {
        "beat_time_s": beat_time,
        "beat_number": torch.floor(beat_time / beat_period),
        "beat_index": beat_index,
        "beat_fraction": beat_fraction,
        "beat_period_s": beat_period,
        "step_size_m": env.step_size.squeeze(1),
        "base_position_m": env.root_states[:, :3] - env.env_origins,
        "roll_rad": roll,
        "pitch_rad": pitch,
        "yaw_rad": yaw,
        "tilt_rad": tilt,
        "body_rate_rad_s": env.base_ang_vel,
        "foot_contact": env._detect_foot_contact(),
        "scheduled_stance": env.scheduled_stance,
        "reference_joint_rad": applied_reference,
        "joint_rad": joint_pos,
        "terminated": env.terminated,
    }
    return {name: value.detach().cpu().numpy().copy() for name, value in record.items()}


def run_rollout(env, runner, rollout_s):
    """Step the robots for rollout_s seconds; return arrays stacked as (T, N, ...)."""
    restart_box_clock(env)
    step_count = int(math.ceil(rollout_s / env.dt))
    records = []
    for _ in range(step_count):
        if runner is not None:
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
        applied_reference = env.gait_reference.clone()
        env.step()
        records.append(record_step(env, applied_reference))
    log = {name: np.stack([r[name] for r in records]) for name in records[0]}
    log["time_s"] = (np.arange(step_count) + 1) * env.dt
    log["dt"] = env.dt
    return log


def find_valid_steps(log):
    """Return (T, N) bool: True before each robot's first termination."""
    terminated = log["terminated"]
    has_fallen = np.cumsum(terminated, axis=0) > 0
    return ~has_fallen


def write_step_log(log, path):
    step_count, env_count = log["terminated"].shape
    header = ["time_s", "env", "beat_number", "beat_index", "beat_fraction"]
    header += ["step_size_m", "base_x_m", "base_y_m", "base_z_m"]
    header += ["roll_rad", "pitch_rad", "yaw_rad", "tilt_rad"]
    header += ["roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s"]
    header += [f"contact_{leg}" for leg in LEG_NAMES]
    header += [f"stance_{leg}" for leg in LEG_NAMES]
    joint_labels = [f"{leg}_{joint}" for leg in LEG_NAMES for joint in JOINT_NAMES]
    header += [f"ref_{label}" for label in joint_labels]
    header += [f"q_{label}" for label in joint_labels]
    header += ["terminated"]
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for step in range(step_count):
            for env_id in range(env_count):
                row = [f"{log['time_s'][step]:.3f}", env_id]
                row += [int(log["beat_number"][step, env_id])]
                row += [int(log["beat_index"][step, env_id])]
                row += [f"{log['beat_fraction'][step, env_id]:.4f}"]
                row += [f"{log['step_size_m'][step, env_id]:.4f}"]
                row += [f"{v:.4f}" for v in log["base_position_m"][step, env_id]]
                for name in ("roll_rad", "pitch_rad", "yaw_rad", "tilt_rad"):
                    row += [f"{log[name][step, env_id]:.4f}"]
                row += [f"{v:.4f}" for v in log["body_rate_rad_s"][step, env_id]]
                row += [int(v) for v in log["foot_contact"][step, env_id]]
                row += [int(v) for v in log["scheduled_stance"][step, env_id]]
                row += [f"{v:.4f}" for v in log["reference_joint_rad"][step, env_id]]
                row += [f"{v:.4f}" for v in log["joint_rad"][step, env_id]]
                row += [int(log["terminated"][step, env_id])]
                writer.writerow(row)


# ------------------------------------------------------
#
# Metrics
#
# ------------------------------------------------------


def list_complete_beats(log, valid):
    """Return (env_id, first_step, end_step, beat_index) for whole, valid beats."""
    beats = []
    beat_number = log["beat_number"]
    for env_id in range(beat_number.shape[1]):
        numbers = beat_number[:, env_id]
        starts = np.flatnonzero(np.diff(numbers) != 0) + 1
        for first_step, end_step in zip(starts[:-1], starts[1:]):
            if valid[first_step:end_step, env_id].all():
                beat_index = int(log["beat_index"][first_step, env_id])
                beats.append((env_id, first_step, end_step, beat_index))
    return beats


def find_stable_steps(log):
    """Return (T, N) bool: the robot is stable at this step (before the hold)."""
    feet_down = log["foot_contact"].all(axis=2)
    level = log["tilt_rad"] < math.radians(STABLE_TILT_DEG)
    roll_pitch_rate = np.abs(log["body_rate_rad_s"][:, :, :2])
    calm = (roll_pitch_rate < STABLE_BODY_RATE_RAD_S).all(axis=2)
    return feet_down & level & calm


def find_touchdowns(log):
    """Return (T, N, 4) bool: the foot touched down at this step."""
    contact = log["foot_contact"]
    previous = np.concatenate((np.ones_like(contact[:1]), contact[:-1]), axis=0)
    return contact & ~previous


def measure_time_to_stability(log, beats):
    """Metric 1: seconds from each hold beat's start to its first stable window."""
    stable = find_stable_steps(log)
    window = int(round(STABLE_HOLD_S / log["dt"]))
    times = []
    for env_id, first_step, end_step, beat_index in beats:
        if reference.BEAT_IS_MOVING[beat_index]:
            continue
        stable_in_beat = stable[first_step:end_step, env_id]
        found = math.nan
        for offset in range(len(stable_in_beat) - window + 1):
            if stable_in_beat[offset : offset + window].all():
                found = offset * log["dt"]
                break
        times.append(found)
    return np.array(times)


def count_scheduled_touchdowns(beat_index):
    return 1 if reference.BEAT_IS_MOVING[beat_index] else 0


def count_foot_repositioning(log, beats):
    """Metric 2: extra touchdowns per pair per beat, as (beats, 2)."""
    touchdowns = find_touchdowns(log)
    extras = []
    for env_id, first_step, end_step, beat_index in beats:
        per_foot = touchdowns[first_step:end_step, env_id].sum(axis=0)
        extra_per_foot = np.maximum(
            per_foot - count_scheduled_touchdowns(beat_index), 0
        )
        extras.append([extra_per_foot[[0, 2]].sum(), extra_per_foot[[1, 3]].sum()])
    return np.array(extras).reshape(-1, 2)


def measure_repositioning_air_time(log, beats):
    """Metric 3: air time after first touchdown, per pair per beat (beats, 2)."""
    contact = log["foot_contact"]
    touchdowns = find_touchdowns(log)
    air_times = []
    for env_id, first_step, end_step, beat_index in beats:
        per_foot = np.zeros(4)
        for foot in range(4):
            foot_contact = contact[first_step:end_step, env_id, foot]
            foot_touchdowns = touchdowns[first_step:end_step, env_id, foot]
            if not reference.BEAT_IS_MOVING[beat_index] and foot_contact[0]:
                anchor = 0
            elif foot_touchdowns.any():
                anchor = int(np.argmax(foot_touchdowns))
            else:
                continue
            per_foot[foot] = (~foot_contact[anchor:]).sum() * log["dt"]
        air_times.append([per_foot[[0, 2]].mean(), per_foot[[1, 3]].mean()])
    return np.array(air_times).reshape(-1, 2)


def measure_step_timing(log, beats, swing_time_s):
    """Metric 4: first touchdown minus scheduled touchdown [s]; NaN if missing."""
    touchdowns = find_touchdowns(log)
    errors = []
    for env_id, first_step, end_step, beat_index in beats:
        if not reference.BEAT_IS_MOVING[beat_index]:
            continue
        beat_period = log["beat_period_s"][first_step, env_id]
        swing_fraction = swing_time_s / beat_period
        left_leads = bool(reference.BEAT_LEADS_WITH_LEFT[beat_index])
        beat_start_time = log["beat_time_s"][first_step, env_id] - (
            log["beat_fraction"][first_step, env_id] * beat_period
        )
        for foot in range(4):
            is_left = foot in (0, 2)
            leads = is_left == left_leads
            scheduled = (1 if leads else 2) * swing_fraction * beat_period
            foot_touchdowns = touchdowns[first_step:end_step, env_id, foot]
            if not foot_touchdowns.any():
                errors.append(math.nan)
                continue
            touchdown_step = first_step + int(np.argmax(foot_touchdowns))
            touchdown_time = log["beat_time_s"][touchdown_step, env_id]
            errors.append(touchdown_time - beat_start_time - scheduled)
    return np.array(errors)


# ------------------------------------------------------
#
# Reference-tracking diagnostics
#
# ------------------------------------------------------


def measure_beat_displacements(log, beats):
    """Body displacement over each moving beat, in the heading at the beat start.

    Returns rows (along-direction m, across-direction m, commanded d m).
    """
    rows = []
    for env_id, first_step, end_step, beat_index in beats:
        if not reference.BEAT_IS_MOVING[beat_index]:
            continue
        start = log["base_position_m"][first_step - 1, env_id, :2]
        end = log["base_position_m"][end_step - 1, env_id, :2]
        yaw = log["yaw_rad"][first_step - 1, env_id]
        world_to_heading = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
        )
        displacement = world_to_heading @ (end - start)
        direction = reference.BEAT_DIRECTIONS[beat_index].numpy()
        across = np.array([-direction[1], direction[0]])
        rows.append(
            [
                displacement @ direction,
                displacement @ across,
                log["step_size_m"][first_step, env_id],
            ]
        )
    return np.array(rows).reshape(-1, 3)


def measure_box_displacements(log, beats):
    """Net world-frame xy displacement over each complete, valid box [m]."""
    displacements = []
    by_env = {}
    for env_id, first_step, end_step, beat_index in beats:
        by_env.setdefault(env_id, []).append((first_step, end_step, beat_index))
    for env_id, env_beats in by_env.items():
        for i in range(len(env_beats) - 5):
            box = env_beats[i : i + 6]
            consecutive = all(a[1] == b[0] for a, b in zip(box[:-1], box[1:]))
            if box[0][2] == 0 and consecutive:
                start = log["base_position_m"][box[0][0] - 1, env_id, :2]
                end = log["base_position_m"][box[-1][1] - 1, env_id, :2]
                displacements.append(np.linalg.norm(end - start))
    return np.array(displacements)


def measure_swing_tracking_error(log, valid):
    """Peak |reference - actual| per joint type over swing feet [rad]."""
    error = np.abs(log["reference_joint_rad"] - log["joint_rad"]).reshape(
        *log["joint_rad"].shape[:2], 4, 3
    )
    swinging = ~log["scheduled_stance"] & valid[:, :, None]
    if not swinging.any():
        return {name: math.nan for name in JOINT_NAMES}
    return {
        name: float(error[..., joint][swinging].max())
        for joint, name in enumerate(JOINT_NAMES)
    }


def describe_falls(log):
    """Return a list of (env_id, time_s, beat_index) for robots that fell."""
    falls = []
    for env_id in range(log["terminated"].shape[1]):
        terminated_steps = np.flatnonzero(log["terminated"][:, env_id])
        if terminated_steps.size:
            step = terminated_steps[0]
            beat_index = int(log["beat_index"][step, env_id])
            falls.append((env_id, float(log["time_s"][step]), beat_index))
    return falls


def summarize(values):
    finite = values[np.isfinite(values)] if values.size else values
    return {
        "count": int(values.size),
        "missing_or_never": int(values.size - finite.size),
        "mean": float(finite.mean()) if finite.size else math.nan,
        "max": float(finite.max()) if finite.size else math.nan,
        "max_abs": float(np.abs(finite).max()) if finite.size else math.nan,
    }


def compute_metrics(log, swing_time_s):
    valid = find_valid_steps(log)
    beats = list_complete_beats(log, valid)
    repositioning = count_foot_repositioning(log, beats)
    air_time = measure_repositioning_air_time(log, beats)
    beat_displacements = measure_beat_displacements(log, beats)
    tilt = np.where(valid, log["tilt_rad"], 0.0)
    metrics = {
        "robots": int(valid.shape[1]),
        "falls (env, time_s, beat 1-6)": [
            (env_id, time_s, beat_index + 1)
            for env_id, time_s, beat_index in describe_falls(log)
        ],
        "complete_valid_beats": len(beats),
        "1_time_to_stability_s": summarize(measure_time_to_stability(log, beats)),
        "2_extra_touchdowns_per_pair_per_beat": {
            pair: summarize(repositioning[:, i]) for i, pair in enumerate(PAIR_NAMES)
        },
        "3_air_time_after_first_touchdown_s": {
            pair: summarize(air_time[:, i]) for i, pair in enumerate(PAIR_NAMES)
        },
        "4_touchdown_minus_scheduled_s": summarize(
            measure_step_timing(log, beats, swing_time_s)
        ),
        "moving_beat_displacement_along_m": summarize(beat_displacements[:, 0]),
        "moving_beat_displacement_across_m": summarize(beat_displacements[:, 1]),
        "commanded_step_size_m": summarize(beat_displacements[:, 2]),
        "net_displacement_per_box_m": summarize(measure_box_displacements(log, beats)),
        "peak_swing_tracking_error_rad": measure_swing_tracking_error(log, valid),
        "max_tilt_deg_before_fall": float(np.degrees(tilt.max())),
    }
    return metrics


def main():
    args = get_evaluation_args()
    with torch.no_grad():
        env, runner, rollout_s = build_env_and_runner(args)
        log = run_rollout(env, runner, rollout_s)
    os.makedirs(os.path.dirname(STEP_LOG_PATH), exist_ok=True)
    write_step_log(log, STEP_LOG_PATH)
    metrics = compute_metrics(log, env.cfg.boxstep.swing_time_s)
    metrics["mode"] = "reference_only" if args.reference_only else "policy"
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {STEP_LOG_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
