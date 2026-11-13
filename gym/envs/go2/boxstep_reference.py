# AI-assisted: box-step reference math (beat clock, step oscillator, foot paths,
# leg inverse kinematics) written from BOXSTEP_MATH.md Parts 1-3 and the function
# table in BOXSTEP_PROJECT_MAP.md, so training and deploy can share one copy.
"""Waltz box-step reference for the Go2.

Pure torch, no simulator imports. Every function is batched over N robots.
Symbols follow BOXSTEP_MATH.md:

    T_b  beat period (s)             k  beat within the box, 0..5
    s    beat fraction, 0 <= s < 1   sigma  swing fraction, T_sw / T_b
    theta step oscillator phase      u  swing progress of a leg pair, 0..1
    q    foot progress over ground   p  body progress over ground
    d    step size (m)               h  standing height, g_c swing clearance

Leg order is FL, FR, RL, RR; joint order is hip, thigh, calf. Pair order is
left, right. Body frame: x forward, y left, z up.
"""

import math

import torch

BEATS_PER_BOX = 6

# Beat schedule, MATH 1.3. Row k is beat k of the box.
BEAT_DIRECTIONS = torch.tensor(
    [
        [1.0, 0.0],  # 0 forward
        [0.0, -1.0],  # 1 right
        [0.0, 0.0],  # 2 hold
        [-1.0, 0.0],  # 3 back
        [0.0, 1.0],  # 4 left
        [0.0, 0.0],  # 5 hold
    ]
)
BEAT_LEADS_WITH_LEFT = torch.tensor([True, False, False, False, True, False])
BEAT_IS_MOVING = torch.tensor([True, True, False, True, True, False])

# Leg geometry from go2.urdf, MATH 3.1.
HIP_OFFSET_M = 0.0955
LINK_LENGTH_M = 0.213
CALF_STRAIGHTEST_RAD = -0.83776
CALF_MOST_BENT_RAD = -2.7227
LEG_SIDE_SIGN = torch.tensor([1.0, -1.0, 1.0, -1.0])
# Which pair (0 = left, 1 = right) each leg belongs to: the pace gait.
LEG_PAIR_INDEX = torch.tensor([0, 1, 0, 1])


def compute_beat_state(time_s, beat_period_s):
    """Return (beat_index (N,) long in 0..5, beat_fraction (N,) in [0, 1))."""
    beats_elapsed = time_s / beat_period_s
    beats_completed = torch.floor(beats_elapsed)
    beat_fraction = beats_elapsed - beats_completed
    beat_index = torch.remainder(beats_completed, BEATS_PER_BOX).long()
    return beat_index, beat_fraction


def compute_step_oscillator(beat_index, beat_fraction, swing_fraction):
    """Return theta (N,) in [0, 2*pi): one cycle across both swings, 0 on holds."""
    is_moving = BEAT_IS_MOVING.to(beat_index.device)[beat_index]
    step_progress = torch.clamp(beat_fraction / (2.0 * swing_fraction), max=1.0)
    step_phase = 2.0 * math.pi * step_progress * is_moving
    return torch.remainder(step_phase, 2.0 * math.pi)


def compute_pair_phases(beat_index, step_phase):
    """Return each pair's oscillator phase (N, 2) in [0, 2*pi), left then right.

    The lead pair's phase is theta and the follow pair's is theta - pi, so
    sin(phase) > 0 exactly while that pair swings. Both are 0 on hold beats.
    """
    is_moving = BEAT_IS_MOVING.to(beat_index.device)[beat_index]
    leads_with_left = BEAT_LEADS_WITH_LEFT.to(beat_index.device)[beat_index]
    follow_phase = step_phase - math.pi * is_moving
    left_phase = torch.where(leads_with_left, step_phase, follow_phase)
    right_phase = torch.where(leads_with_left, follow_phase, step_phase)
    pair_phases = torch.stack((left_phase, right_phase), dim=1)
    return torch.remainder(pair_phases, 2.0 * math.pi)


def compute_box_phase(time_s, beat_period_s):
    """Return the position in the six-beat box (N,) as an angle in [0, 2*pi)."""
    boxes_elapsed = time_s / (BEATS_PER_BOX * beat_period_s)
    return 2.0 * math.pi * torch.remainder(boxes_elapsed, 1.0)


def compute_swing_progress(beat_index, beat_fraction, swing_fraction):
    """Return u (N, 2) in [0, 1] for the left and right pair. Zero on holds."""
    is_moving = BEAT_IS_MOVING.to(beat_index.device)[beat_index]
    leads_with_left = BEAT_LEADS_WITH_LEFT.to(beat_index.device)[beat_index]
    lead_progress = torch.clamp(beat_fraction / swing_fraction, 0.0, 1.0)
    follow_progress = torch.clamp(
        (beat_fraction - swing_fraction) / swing_fraction, 0.0, 1.0
    )
    left_progress = torch.where(leads_with_left, lead_progress, follow_progress)
    right_progress = torch.where(leads_with_left, follow_progress, lead_progress)
    swing_progress = torch.stack((left_progress, right_progress), dim=1)
    return swing_progress * is_moving.unsqueeze(1)


def compute_body_progress(beat_fraction, swing_fraction):
    """Return p (N,) in [0, 1]: option B, rest-to-rest over the two swings."""
    window_progress = torch.clamp(beat_fraction / (2.0 * swing_fraction), max=1.0)
    return 0.5 * (1.0 - torch.cos(math.pi * window_progress))


def compute_foot_offsets(
    beat_index, beat_fraction, swing_fraction, step_size_m, swing_clearance_m
):
    """Return each foot's offset from neutral (N, 4, 3) in metres.

    Horizontal: r = d * (q - p) * e_k. Vertical: g_c * sin(pi * u).
    """
    pair_progress = compute_swing_progress(beat_index, beat_fraction, swing_fraction)
    foot_progress = pair_progress[:, LEG_PAIR_INDEX.to(beat_index.device)]
    foot_ground_progress = 0.5 * (1.0 - torch.cos(math.pi * foot_progress))
    body_progress = compute_body_progress(beat_fraction, swing_fraction)
    direction = BEAT_DIRECTIONS.to(beat_fraction.device)[beat_index]

    relative_progress = foot_ground_progress - body_progress.unsqueeze(1)
    horizontal_offset = (
        step_size_m.reshape(-1, 1, 1)
        * relative_progress.unsqueeze(2)
        * direction.unsqueeze(1)
    )
    lift = swing_clearance_m * torch.sin(math.pi * foot_progress)
    return torch.cat((horizontal_offset, lift.unsqueeze(2)), dim=2)


def compute_scheduled_stance(beat_index, beat_fraction, swing_fraction):
    """Return (N, 4) bool, True where a foot is scheduled to be on the ground."""
    is_moving = BEAT_IS_MOVING.to(beat_index.device)[beat_index]
    leads_with_left = BEAT_LEADS_WITH_LEFT.to(beat_index.device)[beat_index]
    lead_swinging = is_moving & (beat_fraction < swing_fraction)
    follow_swinging = (
        is_moving
        & (beat_fraction >= swing_fraction)
        & (beat_fraction < 2.0 * swing_fraction)
    )
    left_swinging = torch.where(leads_with_left, lead_swinging, follow_swinging)
    right_swinging = torch.where(leads_with_left, follow_swinging, lead_swinging)
    pair_swinging = torch.stack((left_swinging, right_swinging), dim=1)
    return ~pair_swinging[:, LEG_PAIR_INDEX.to(beat_index.device)]


def compute_body_velocity(
    beat_index, beat_fraction, swing_fraction, step_size_m, beat_period_s
):
    """Return the reference body velocity d * dp/dt * e_k (N, 2) in m/s."""
    window = 2.0 * swing_fraction
    in_window = beat_fraction < window
    progress_rate_per_beat = (
        (math.pi / (2.0 * window)) * torch.sin(math.pi * beat_fraction / window)
    ) * in_window
    speed = step_size_m * progress_rate_per_beat / beat_period_s
    direction = BEAT_DIRECTIONS.to(beat_fraction.device)[beat_index]
    return speed.unsqueeze(1) * direction


def compute_neutral_foot_positions(standing_height_m, device="cpu"):
    """Return each foot's neutral position (4, 3) in its own hip frame.

    Directly below the thigh joint at the standing height: (0, side * l1, -h).
    """
    side_sign = LEG_SIDE_SIGN.to(device)
    return torch.stack(
        (
            torch.zeros(4, device=device),
            side_sign * HIP_OFFSET_M,
            torch.full((4,), -standing_height_m, device=device),
        ),
        dim=1,
    )


def solve_leg_joint_angles(foot_position_m):
    """Return joint angles (N, 12) for foot targets (N, 4, 3) in each hip frame.

    MATH 3.3-3.5. The knee is clamped to its joint limits, which also caps the
    leg at the reach limit; the leg then points at the target but falls short.
    """
    foot_x, foot_y, foot_z = foot_position_m.unbind(dim=2)
    side_offset = LEG_SIDE_SIGN.to(foot_position_m.device) * HIP_OFFSET_M

    leg_plane_length = torch.sqrt(
        torch.clamp(foot_y**2 + foot_z**2 - HIP_OFFSET_M**2, min=0.0)
    )
    hip_angle = torch.atan2(
        leg_plane_length * foot_y + side_offset * foot_z,
        side_offset * foot_y - leg_plane_length * foot_z,
    )

    thigh_to_foot_squared = foot_x**2 + leg_plane_length**2
    knee_cosine = torch.clamp(
        thigh_to_foot_squared / (2.0 * LINK_LENGTH_M**2) - 1.0,
        min=math.cos(CALF_MOST_BENT_RAD),
        max=math.cos(CALF_STRAIGHTEST_RAD),
    )
    calf_angle = -torch.arccos(knee_cosine)
    thigh_angle = torch.atan2(-foot_x, leg_plane_length) - 0.5 * calf_angle

    joint_angles = torch.stack((hip_angle, thigh_angle, calf_angle), dim=2)
    return joint_angles.reshape(foot_position_m.shape[0], 12)


def compute_foot_positions(joint_angles):
    """Return each foot's position (N, 4, 3) in its hip frame from (N, 12) angles.

    Forward kinematics, MATH 3.2. The inverse of solve_leg_joint_angles.
    """
    hip, thigh, calf = joint_angles.reshape(-1, 4, 3).unbind(dim=2)
    side_offset = LEG_SIDE_SIGN.to(joint_angles.device) * HIP_OFFSET_M
    foot_x = -LINK_LENGTH_M * (torch.sin(thigh) + torch.sin(thigh + calf))
    plane_z = -LINK_LENGTH_M * (torch.cos(thigh) + torch.cos(thigh + calf))
    foot_y = side_offset * torch.cos(hip) - plane_z * torch.sin(hip)
    foot_z = side_offset * torch.sin(hip) + plane_z * torch.cos(hip)
    return torch.stack((foot_x, foot_y, foot_z), dim=2)


def compute_reference_foot_positions(
    time_s,
    beat_period_s,
    swing_time_s,
    step_size_m,
    standing_height_m,
    swing_clearance_m,
):
    """Return the reference foot targets (N, 4, 3), each in its own hip frame."""
    beat_index, beat_fraction = compute_beat_state(time_s, beat_period_s)
    swing_fraction = swing_time_s / beat_period_s
    foot_offsets = compute_foot_offsets(
        beat_index, beat_fraction, swing_fraction, step_size_m, swing_clearance_m
    )
    neutral_feet = compute_neutral_foot_positions(standing_height_m, time_s.device)
    return neutral_feet.unsqueeze(0) + foot_offsets


def compute_reference_joint_angles(
    time_s,
    beat_period_s,
    swing_time_s,
    step_size_m,
    standing_height_m,
    swing_clearance_m,
):
    """Return the absolute reference joint angles (N, 12) for the whole robot.

    time_s, beat_period_s and step_size_m are (N,) tensors; swing_time_s,
    standing_height_m and swing_clearance_m are floats.
    """
    foot_targets = compute_reference_foot_positions(
        time_s,
        beat_period_s,
        swing_time_s,
        step_size_m,
        standing_height_m,
        swing_clearance_m,
    )
    return solve_leg_joint_angles(foot_targets)
