# AI-assisted: Go2BoxStep task (buffers, beat clock, reference, observations,
# rewards) written from BOXSTEP_PROJECT_MAP.md on top of Go2Trot.
import torch

from gym.envs.go2 import boxstep_reference as reference
from gym.envs.go2.go2trot import Go2Trot
from gym.utils.sampling import masked_update, torch_rand_float


class Go2BoxStep(Go2Trot):
    """Waltz box step in a pace gait, driven by a beat clock.

    The reference supplies absolute joint angles in gait_reference (default joint
    angles are zero), and the policy adds dof_pos_target corrections, as in
    Go2Trot._compute_torques.
    """

    def _init_buffers(self):
        # Go2Trot._init_buffers ends by calling the phase-observation and
        # gait-reference updates, which read these buffers.
        self._create_boxstep_buffers()
        super()._init_buffers()

    def _create_boxstep_buffers(self):
        settings = self.cfg.boxstep

        def create_column(width=1, fill=0.0, dtype=torch.float):
            return torch.full(
                (self.num_envs, width), fill, dtype=dtype, device=self.device
            )

        nominal_tempo_bpm = sum(settings.tempo_bpm_range) / 2.0
        nominal_step_size_m = sum(settings.step_size_range) / 2.0
        self.beat_time = create_column()
        self.beat_period = create_column(fill=60.0 / nominal_tempo_bpm)
        self.step_size = create_column(fill=nominal_step_size_m)
        self.step_oscillator_obs = create_column(4)
        self.box_phase_obs = create_column(2)
        self.scheduled_stance = create_column(4, True, torch.bool)
        self.reference_body_velocity = create_column(2)
        self.touchdown_count = create_column(4)
        self.previous_foot_contact = create_column(4, True, torch.bool)
        self.new_touchdown = create_column(4, False, torch.bool)

        neutral_feet = reference.compute_neutral_foot_positions(
            settings.standing_height_m, self.device
        )
        standing_angles = reference.solve_leg_joint_angles(neutral_feet.unsqueeze(0))
        self._standing_dof_pos = torch.zeros(1, self.num_dof, device=self.device)
        self._standing_dof_pos[:, self.actuated_dof_indices] = standing_angles

    # ------------------------------------------------------
    #
    # Clock and reference
    #
    # ------------------------------------------------------

    def _compute_beat_state(self):
        """Return (beat_index, beat_fraction, swing_fraction), each (N,)."""
        beat_period = self.beat_period.squeeze(1)
        beat_index, beat_fraction = reference.compute_beat_state(
            self.beat_time.squeeze(1), beat_period
        )
        swing_fraction = self.cfg.boxstep.swing_time_s / beat_period
        return beat_index, beat_fraction, swing_fraction

    def _advance_phase(self):
        # Runs once per physics substep, like Go2Trot._advance_phase.
        self.beat_time.add_(self.dt / self.cfg.control.decimation)
        _, beat_fraction, _ = self._compute_beat_state()
        self.phase[:] = 2 * torch.pi * beat_fraction.unsqueeze(1)

    def _update_gait_reference(self):
        settings = self.cfg.boxstep
        beat_index, beat_fraction, swing_fraction = self._compute_beat_state()
        step_size = self.step_size.squeeze(1)
        self.gait_reference[:] = reference.compute_reference_joint_angles(
            self.beat_time.squeeze(1),
            self.beat_period.squeeze(1),
            settings.swing_time_s,
            step_size,
            settings.standing_height_m,
            settings.swing_clearance_m,
        )
        self.scheduled_stance[:] = reference.compute_scheduled_stance(
            beat_index, beat_fraction, swing_fraction
        )
        self.reference_body_velocity[:] = reference.compute_body_velocity(
            beat_index,
            beat_fraction,
            swing_fraction,
            step_size,
            self.beat_period.squeeze(1),
        )

    def _update_phase_observation(self):
        super()._update_phase_observation()
        beat_index, beat_fraction, swing_fraction = self._compute_beat_state()
        step_phase = reference.compute_step_oscillator(
            beat_index, beat_fraction, swing_fraction
        )
        pair_phases = reference.compute_pair_phases(beat_index, step_phase)
        # Order: sin left, cos left, sin right, cos right.
        self.step_oscillator_obs[:] = torch.stack(
            (torch.sin(pair_phases), torch.cos(pair_phases)), dim=2
        ).reshape(self.num_envs, 4)
        box_phase = reference.compute_box_phase(
            self.beat_time.squeeze(1), self.beat_period.squeeze(1)
        )
        self.box_phase_obs[:] = torch.stack(
            (torch.sin(box_phase), torch.cos(box_phase)), dim=1
        )

    # ------------------------------------------------------
    #
    # Reset and commands
    #
    # ------------------------------------------------------

    def reset_to_basic(self, reset_mask):
        """Reset to the standing pose instead of Go2Trot's straight legs."""
        super().reset_to_basic(reset_mask)
        masked_update(self.dof_pos, self._standing_dof_pos, reset_mask)

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        settings = self.cfg.boxstep
        tempo_bpm = torch_rand_float(
            *settings.tempo_bpm_range, (self.num_envs, 1), device=self.device
        )
        step_size = torch_rand_float(
            *settings.step_size_range, (self.num_envs, 1), device=self.device
        )
        starting_beat = torch.randint(
            reference.BEATS_PER_BOX, (self.num_envs, 1), device=self.device
        )
        beat_period = 60.0 / tempo_bpm
        masked_update(self.beat_period, beat_period, reset_mask)
        masked_update(self.step_size, step_size, reset_mask)
        masked_update(self.beat_time, starting_beat * beat_period, reset_mask)
        masked_update(self.phase_frequency, tempo_bpm / 60.0, reset_mask)
        # Episodes start on a beat boundary, so the beat fraction is zero.
        self.phase.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.touchdown_count.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.previous_foot_contact.masked_fill_(reset_mask.unsqueeze(1), True)
        self.new_touchdown.masked_fill_(reset_mask.unsqueeze(1), False)

    def _resample_commands(self, command_mask):
        """No velocity commands in the box step: commands stay zero."""
        return None

    # ------------------------------------------------------
    #
    # Contacts
    #
    # ------------------------------------------------------

    def _post_decimation_step(self):
        super()._post_decimation_step()
        # Refresh the schedule at the new time so rewards and contacts agree.
        self._update_gait_reference()
        self._count_touchdowns()

    def _detect_foot_contact(self):
        """Return (N, 4) bool: vertical foot force above the contact threshold."""
        vertical_force = self.contact_forces[:, self.feet_indices, 2]
        return vertical_force > self.cfg.reward_settings.foot_contact_force_n

    def _count_touchdowns(self):
        foot_contact = self._detect_foot_contact()
        self.new_touchdown[:] = foot_contact & ~self.previous_foot_contact
        _, beat_fraction, _ = self._compute_beat_state()
        # The first control step of each beat clears that beat's count.
        starts_new_beat = beat_fraction * self.beat_period.squeeze(1) < self.dt
        self.touchdown_count.masked_fill_(starts_new_beat.unsqueeze(1), 0.0)
        self.touchdown_count.add_(self.new_touchdown.float())
        self.previous_foot_contact[:] = foot_contact

    def _count_allowed_touchdowns(self):
        """Return (N, 1): one scheduled touchdown per foot on moving beats."""
        beat_index, _, _ = self._compute_beat_state()
        is_moving = reference.BEAT_IS_MOVING.to(self.device)[beat_index]
        return is_moving.float().unsqueeze(1)

    # ------------------------------------------------------
    #
    # Rewards
    #
    # ------------------------------------------------------

    def _reward_scheduled_support(self):
        """Fraction of the feet scheduled as stance that are in contact."""
        supporting = self.scheduled_stance & self._detect_foot_contact()
        return supporting.sum(dim=1) / self.scheduled_stance.sum(dim=1).clamp(min=1)

    def _reward_swing_contact(self):
        """Penalize load carried by feet scheduled as swing."""
        scheduled_swing = (~self.scheduled_stance).float()
        return -torch.mean(scheduled_swing * self._foot_contact_strength(), dim=1)

    def _reward_tracking_body_velocity(self):
        """Body velocity follows reference_body_velocity (body frame, xy)."""
        error = self.reference_body_velocity - self.base_lin_vel[:, :2]
        width = self.cfg.boxstep.velocity_tracking_width
        return torch.exp(-torch.sum(torch.square(error), dim=1) / width**2)

    def _reward_stance_foot_slip(self):
        """Penalize horizontal speed of planted feet (scheduled stance, in contact)."""
        planted = self.scheduled_stance & self._detect_foot_contact()
        foot_speed_squared = torch.sum(
            torch.square(self._rigid_body_lin_vel[:, self.feet_indices, :2]), dim=2
        )
        return -torch.sum(planted * foot_speed_squared, dim=1)

    def _reward_extra_touchdowns(self):
        """Penalize each touchdown beyond the ones scheduled in this beat."""
        extra = self.new_touchdown & (
            self.touchdown_count > self._count_allowed_touchdowns()
        )
        return -extra.sum(dim=1).float()

    def _reward_correction_size(self):
        """Penalize large policy corrections, in units of the action scale."""
        normalized = self.dof_pos_target / self.scales["dof_pos_target"]
        return -torch.mean(torch.square(normalized), dim=1)
