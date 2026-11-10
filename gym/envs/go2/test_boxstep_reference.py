# AI-assisted: pytest versions of the sanity-check table in BOXSTEP_PROJECT_MAP.md,
# plus the forward kinematics of BOXSTEP_MATH.md 3.2 used as the round-trip oracle.
import math

import pytest
import torch

from gym.envs.go2 import boxstep_reference as reference

BEAT_PERIOD_S = 60.0 / 105.0
SWING_TIME_S = 0.2
STANDING_HEIGHT_M = 0.30
SWING_CLEARANCE_M = 0.06
REACH_LIMIT_M = 2.0 * reference.LINK_LENGTH_M * math.cos(0.83776 / 2.0)


def compute_foot_positions(joint_angles):
    """Forward kinematics, MATH 3.2: joint angles (N, 12) -> feet (N, 4, 3)."""
    hip, thigh, calf = joint_angles.reshape(-1, 4, 3).unbind(dim=2)
    link = reference.LINK_LENGTH_M
    foot_x = -link * (torch.sin(thigh) + torch.sin(thigh + calf))
    plane_z = -link * (torch.cos(thigh) + torch.cos(thigh + calf))
    plane_y = reference.LEG_SIDE_SIGN * reference.HIP_OFFSET_M
    foot_y = plane_y * torch.cos(hip) - plane_z * torch.sin(hip)
    foot_z = plane_y * torch.sin(hip) + plane_z * torch.cos(hip)
    return torch.stack((foot_x, foot_y, foot_z), dim=2)


def sample_whole_box(beat_period_s, samples_per_beat=2000):
    """Return times (N,) covering one box, and matching (N,) beat periods."""
    count = reference.BEATS_PER_BOX * samples_per_beat
    time_s = torch.arange(count, dtype=torch.float64) * beat_period_s / samples_per_beat
    return time_s, torch.full_like(time_s, beat_period_s)


def compute_box_offsets(step_size_m, beat_period_s=BEAT_PERIOD_S):
    time_s, beat_period = sample_whole_box(beat_period_s)
    beat_index, beat_fraction = reference.compute_beat_state(time_s, beat_period)
    swing_fraction = SWING_TIME_S / beat_period
    step_size = torch.full_like(time_s, step_size_m)
    return reference.compute_foot_offsets(
        beat_index, beat_fraction, swing_fraction, step_size, SWING_CLEARANCE_M
    )


@pytest.mark.parametrize("beat_fraction_value", [0.0, 1.0 - 1e-9])
def test_foot_offsets_are_zero_at_start_and_end_of_every_beat(beat_fraction_value):
    beat_index = torch.arange(6)
    beat_fraction = torch.full((6,), beat_fraction_value, dtype=torch.float64)
    swing_fraction = SWING_TIME_S / torch.full((6,), BEAT_PERIOD_S, dtype=torch.float64)
    offsets = reference.compute_foot_offsets(
        beat_index,
        beat_fraction,
        swing_fraction,
        torch.full((6,), 0.30, dtype=torch.float64),
        SWING_CLEARANCE_M,
    )
    torch.testing.assert_close(offsets, torch.zeros_like(offsets), atol=1e-9, rtol=0)


def test_forward_then_inverse_kinematics_returns_the_same_angles():
    generator = torch.Generator().manual_seed(0)
    count = 20000

    def sample_uniform(low, high):
        return low + (high - low) * torch.rand(
            count, 4, generator=generator, dtype=torch.float64
        )

    joint_angles = torch.stack(
        (
            sample_uniform(-0.8, 0.8),
            sample_uniform(-0.5, 1.5),
            sample_uniform(-2.6, -0.9),
        ),
        dim=2,
    ).reshape(count, 12)
    feet = compute_foot_positions(joint_angles)
    # The formulas hold only for a foot below the hip axis in the leg's own
    # plane, z' < 0 (MATH 3.6). Keep a 5 cm margin away from that boundary.
    thigh = joint_angles.reshape(count, 4, 3)[:, :, 1]
    calf = joint_angles.reshape(count, 4, 3)[:, :, 2]
    plane_z = -reference.LINK_LENGTH_M * (torch.cos(thigh) + torch.cos(thigh + calf))
    below_hip = (plane_z < -0.05).all(dim=1)
    assert below_hip.sum() > 1000
    recovered = reference.solve_leg_joint_angles(feet[below_hip])
    torch.testing.assert_close(recovered, joint_angles[below_hip], atol=1e-7, rtol=0)


def test_forward_kinematics_matches_math_notes_default_pose():
    default_pose = torch.tensor(4 * [0.0, 0.66, -1.36], dtype=torch.float64)
    left_front_foot = compute_foot_positions(default_pose.unsqueeze(0))[0, 0]
    expected = torch.tensor([0.007, 0.0955, -0.331], dtype=torch.float64)
    torch.testing.assert_close(left_front_foot, expected, atol=1e-3, rtol=0)


def test_reference_angles_put_feet_on_the_commanded_targets():
    time_s, beat_period = sample_whole_box(BEAT_PERIOD_S, samples_per_beat=200)
    step_size = torch.full_like(time_s, 0.30)
    joint_angles = reference.compute_reference_joint_angles(
        time_s,
        beat_period,
        SWING_TIME_S,
        step_size,
        STANDING_HEIGHT_M,
        SWING_CLEARANCE_M,
    )
    beat_index, beat_fraction = reference.compute_beat_state(time_s, beat_period)
    targets = reference.compute_neutral_foot_positions(STANDING_HEIGHT_M).to(
        torch.float64
    ) + reference.compute_foot_offsets(
        beat_index,
        beat_fraction,
        SWING_TIME_S / beat_period,
        step_size,
        SWING_CLEARANCE_M,
    )
    torch.testing.assert_close(
        compute_foot_positions(joint_angles), targets, atol=1e-9, rtol=0
    )


@pytest.mark.parametrize("tempo_bpm", [89.0, 105.0, 121.0])
def test_left_and_right_pairs_never_swing_at_the_same_time(tempo_bpm):
    beat_period_s = 60.0 / tempo_bpm
    time_s, beat_period = sample_whole_box(beat_period_s)
    beat_index, beat_fraction = reference.compute_beat_state(time_s, beat_period)
    swing_fraction = SWING_TIME_S / beat_period
    stance = reference.compute_scheduled_stance(
        beat_index, beat_fraction, swing_fraction
    )
    left_swinging = ~stance[:, 0]
    right_swinging = ~stance[:, 1]
    assert not (left_swinging & right_swinging).any()
    # Each leg pair agrees within itself (pace gait).
    assert torch.equal(stance[:, 0], stance[:, 2])
    assert torch.equal(stance[:, 1], stance[:, 3])
    # The swing windows of u do not overlap either.
    swing_progress = reference.compute_swing_progress(
        beat_index, beat_fraction, swing_fraction
    )
    left_moving = (swing_progress[:, 0] > 0) & (swing_progress[:, 0] < 1)
    right_moving = (swing_progress[:, 1] > 0) & (swing_progress[:, 1] < 1)
    assert not (left_moving & right_moving).any()


def integrate_body_displacement(time_s, beat_period, step_size_m):
    beat_index, beat_fraction = reference.compute_beat_state(time_s, beat_period)
    velocity = reference.compute_body_velocity(
        beat_index,
        beat_fraction,
        SWING_TIME_S / beat_period,
        torch.full_like(time_s, step_size_m),
        beat_period,
    )
    return torch.trapezoid(velocity, time_s, dim=0)


@pytest.mark.parametrize("beat", [0, 1, 3, 4])
def test_body_displacement_over_one_moving_beat_is_the_step_size(beat):
    step_size_m = 0.25
    samples = 20001
    time_s = (
        beat + torch.linspace(0.0, 1.0 - 1e-12, samples, dtype=torch.float64)
    ) * BEAT_PERIOD_S
    beat_period = torch.full_like(time_s, BEAT_PERIOD_S)
    displacement = integrate_body_displacement(time_s, beat_period, step_size_m)
    expected = step_size_m * reference.BEAT_DIRECTIONS[beat].to(torch.float64)
    torch.testing.assert_close(displacement, expected, atol=1e-6, rtol=0)


def test_body_displacement_over_all_six_beats_is_zero():
    time_s, beat_period = sample_whole_box(BEAT_PERIOD_S, samples_per_beat=20000)
    displacement = integrate_body_displacement(time_s, beat_period, 0.30)
    torch.testing.assert_close(
        displacement, torch.zeros(2, dtype=torch.float64), atol=1e-6, rtol=0
    )


def test_farthest_thigh_to_foot_distance_at_largest_step_is_within_reach():
    offsets = compute_box_offsets(step_size_m=0.30)
    feet = (
        reference.compute_neutral_foot_positions(STANDING_HEIGHT_M).to(torch.float64)
        + offsets
    )
    foot_x, foot_y, foot_z = feet.unbind(dim=2)
    leg_plane_length_squared = foot_y**2 + foot_z**2 - reference.HIP_OFFSET_M**2
    thigh_to_foot = torch.sqrt(foot_x**2 + leg_plane_length_squared)
    assert thigh_to_foot.max() < REACH_LIMIT_M
    # MATH 3.7 table: outward side step at d = 0.30, h = 0.30 reaches 0.376 m.
    assert thigh_to_foot.max() == pytest.approx(0.376, abs=2e-3)


def test_hold_beats_have_zero_offsets_and_all_feet_in_stance():
    time_s, beat_period = sample_whole_box(BEAT_PERIOD_S)
    beat_index, beat_fraction = reference.compute_beat_state(time_s, beat_period)
    swing_fraction = SWING_TIME_S / beat_period
    offsets = reference.compute_foot_offsets(
        beat_index,
        beat_fraction,
        swing_fraction,
        torch.full_like(time_s, 0.30),
        SWING_CLEARANCE_M,
    )
    stance = reference.compute_scheduled_stance(
        beat_index, beat_fraction, swing_fraction
    )
    oscillator = reference.compute_step_oscillator(
        beat_index, beat_fraction, swing_fraction
    )
    hold = (beat_index == 2) | (beat_index == 5)
    assert hold.any()
    assert torch.equal(offsets[hold], torch.zeros_like(offsets[hold]))
    assert stance[hold].all()
    assert torch.equal(oscillator[hold], torch.zeros_like(oscillator[hold]))


def test_zero_step_size_steps_in_place():
    offsets = compute_box_offsets(step_size_m=0.0)
    assert torch.equal(offsets[:, :, :2], torch.zeros_like(offsets[:, :, :2]))


def test_standing_pose_puts_feet_below_thigh_joints_at_standing_height():
    neutral = reference.compute_neutral_foot_positions(STANDING_HEIGHT_M)
    standing_angles = reference.solve_leg_joint_angles(neutral.unsqueeze(0))
    torch.testing.assert_close(
        compute_foot_positions(standing_angles.double())[0],
        neutral.double(),
        atol=1e-6,
        rtol=0,
    )
    assert torch.all(standing_angles.reshape(4, 3)[:, 0] == 0.0)


def test_pair_phases_have_positive_sine_exactly_while_that_pair_swings():
    time_s, beat_period = sample_whole_box(BEAT_PERIOD_S)
    beat_index, beat_fraction = reference.compute_beat_state(time_s, beat_period)
    swing_fraction = SWING_TIME_S / beat_period
    step_phase = reference.compute_step_oscillator(
        beat_index, beat_fraction, swing_fraction
    )
    pair_phases = reference.compute_pair_phases(beat_index, step_phase)
    stance = reference.compute_scheduled_stance(
        beat_index, beat_fraction, swing_fraction
    )
    pair_swinging = ~stance[:, :2]
    # Skip the lift-off and touchdown instants, where sin(phase) is zero.
    clear_of_edges = torch.sin(pair_phases).abs() > 1e-6
    swinging_by_phase = torch.sin(pair_phases) > 0
    assert torch.equal(swinging_by_phase[clear_of_edges], pair_swinging[clear_of_edges])
