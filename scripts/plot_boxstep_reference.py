# AI-assisted: plots of the box-step reference (theta, q, p, foot offsets, joint
# angles) over one six-beat box, straight from boxstep_reference with no simulator.
"""Plot the box-step reference for one box.

Usage:
    uv run --frozen scripts/plot_boxstep_reference.py \
        [--tempo_bpm 105] [--step_size 0.25]

Writes logs/boxstep_reference.pdf and logs/boxstep_reference.png (gitignored).
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gym.envs.go2 import boxstep_reference as reference  # noqa: E402
from gym.envs.go2.go2boxstep_config import Go2BoxStepCfg  # noqa: E402

FIGURE_STEM = "logs/boxstep_reference"
SAMPLES_PER_BEAT = 400
LEG_NAMES = ["FL", "FR", "RL", "RR"]
JOINT_NAMES = ["hip", "thigh", "calf"]


def compute_reference_trace(tempo_bpm, step_size_m, settings):
    """Return a dict of reference signals sampled over one box, time along axis 0."""
    beat_period_s = 60.0 / tempo_bpm
    sample_count = reference.BEATS_PER_BOX * SAMPLES_PER_BEAT
    time_s = torch.arange(sample_count, dtype=torch.float64) * (
        beat_period_s / SAMPLES_PER_BEAT
    )
    beat_period = torch.full_like(time_s, beat_period_s)
    step_size = torch.full_like(time_s, step_size_m)
    swing_fraction = settings.swing_time_s / beat_period

    beat_index, beat_fraction = reference.compute_beat_state(time_s, beat_period)
    swing_progress = reference.compute_swing_progress(
        beat_index, beat_fraction, swing_fraction
    )
    trace = {
        "time_s": time_s,
        "beat_period_s": beat_period_s,
        "step_phase": reference.compute_step_oscillator(
            beat_index, beat_fraction, swing_fraction
        ),
        "foot_progress": 0.5 * (1.0 - torch.cos(math.pi * swing_progress)),
        "body_progress": reference.compute_body_progress(beat_fraction, swing_fraction)
        * reference.BEAT_IS_MOVING[beat_index],
        "foot_offsets": reference.compute_foot_offsets(
            beat_index,
            beat_fraction,
            swing_fraction,
            step_size,
            settings.swing_clearance_m,
        ),
        "scheduled_stance": reference.compute_scheduled_stance(
            beat_index, beat_fraction, swing_fraction
        ),
        "joint_angles": reference.compute_reference_joint_angles(
            time_s,
            beat_period,
            settings.swing_time_s,
            step_size,
            settings.standing_height_m,
            settings.swing_clearance_m,
        ),
    }
    return trace


def mark_beat_boundaries(axis, beat_period_s):
    for beat in range(reference.BEATS_PER_BOX + 1):
        axis.axvline(beat * beat_period_s, color="#898781", linestyle="--", lw=0.6)


def plot_reference_trace(trace, tempo_bpm, step_size_m):
    time_s = trace["time_s"].numpy()
    figure, axes = plt.subplots(6, 1, sharex=True, figsize=(10, 14))
    (
        oscillator_axis,
        progress_axis,
        left_axis,
        right_axis,
        stance_axis,
        joint_axis,
    ) = axes

    oscillator_axis.plot(time_s, trace["step_phase"].numpy(), label="theta")
    oscillator_axis.set_ylabel("step oscillator [rad]")
    oscillator_axis.set_title(
        f"Box-step reference: {tempo_bpm:.0f} BPM, d = {step_size_m:.2f} m "
        "(dashed lines are beat boundaries; beats 3 and 6 are holds)"
    )

    progress_axis.plot(time_s, trace["foot_progress"][:, 0].numpy(), label="q left")
    progress_axis.plot(time_s, trace["foot_progress"][:, 1].numpy(), label="q right")
    progress_axis.plot(
        time_s, trace["body_progress"].numpy(), color="k", ls=":", label="p body"
    )
    progress_axis.set_ylabel("progress over ground [-]")

    for axis, leg, pair_name in ((left_axis, 0, "left"), (right_axis, 1, "right")):
        offsets = trace["foot_offsets"][:, leg].numpy()
        for component, name in enumerate(["x", "y", "z"]):
            axis.plot(time_s, offsets[:, component], label=f"{name} offset")
        axis.set_ylabel(f"{pair_name} pair\noffset [m]")

    stance = trace["scheduled_stance"].numpy()
    stance_axis.plot(time_s, stance[:, 0] + 0.02, label="left pair stance")
    stance_axis.plot(time_s, stance[:, 1] - 0.02, label="right pair stance")
    stance_axis.set_ylabel("scheduled stance")

    joint_angles = trace["joint_angles"].reshape(-1, 4, 3).numpy()
    for leg in (0, 1):  # RL equals FL and RR equals FR in the hip frame.
        for joint, joint_name in enumerate(JOINT_NAMES):
            joint_axis.plot(
                time_s,
                joint_angles[:, leg, joint],
                ls="-" if leg == 0 else "--",
                label=f"{LEG_NAMES[leg]} {joint_name}",
            )
    joint_axis.set_ylabel("reference angle [rad]")
    joint_axis.set_xlabel("time [s]")

    for axis in axes:
        mark_beat_boundaries(axis, trace["beat_period_s"])
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right", fontsize=7, ncol=3)
    figure.tight_layout()
    return figure


def main():
    settings = Go2BoxStepCfg.boxstep
    parser = argparse.ArgumentParser(description="Plot one box of the reference.")
    parser.add_argument("--tempo_bpm", type=float, default=settings.tempo_bpm_range[0])
    parser.add_argument("--step_size", type=float, default=settings.step_size_range[0])
    args = parser.parse_args()

    trace = compute_reference_trace(args.tempo_bpm, args.step_size, settings)
    figure = plot_reference_trace(trace, args.tempo_bpm, args.step_size)
    os.makedirs(os.path.dirname(FIGURE_STEM), exist_ok=True)
    figure.savefig(FIGURE_STEM + ".pdf")
    figure.savefig(FIGURE_STEM + ".png", dpi=110)
    print(f"Wrote {FIGURE_STEM}.pdf and {FIGURE_STEM}.png")


if __name__ == "__main__":
    main()
