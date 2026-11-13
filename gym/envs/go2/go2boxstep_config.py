# AI-assisted: box-step task and runner configs built from BOXSTEP_PLAN.md and the
# config section of BOXSTEP_PROJECT_MAP.md. Reward weights and the velocity
# tracking width are provisional starting values, not decisions by the user.
from gym.envs.go2.go2_config import BASE_HEIGHT_REF
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg

STANDING_HEIGHT_M = 0.30  # hip joint above the foot centres, decided 2026-11-10
FOOT_RADIUS_M = 0.022  # go2.urdf foot collision sphere


class Go2BoxStepCfg(Go2TrotCfg):
    class env(Go2TrotCfg.env):
        episode_length_s = 10  # about three boxes at 105 BPM

    class init_state(Go2TrotCfg.init_state):
        # Robots reset in the standing pose (leg IK of the neutral feet). The hip
        # joints sit level with the base origin in go2.urdf, so the base starts
        # at the standing height plus the foot radius, feet touching the floor.
        pos = [0.0, 0.0, STANDING_HEIGHT_M + FOOT_RADIUS_M]
        reset_mode = "reset_to_basic"

    class boxstep:
        tempo_bpm_range = [105.0, 105.0]  # later [89, 121]
        step_size_range = [0.25, 0.30]  # d [m]
        swing_time_s = 0.2  # T_sw, one leg pair's swing
        standing_height_m = STANDING_HEIGHT_M  # h
        swing_clearance_m = 0.06  # g_c
        # Provisional starting value (not decided by the user): width of the
        # exp(-|v_ref - v|^2 / width^2) body-velocity reward [m/s]. Reference
        # peak speed is about 1 m/s at d = 0.25 m, so 0.3 m/s is tighter than
        # go2trot's tracking_sigma = 0.25 (an effective width of 0.5 m/s).
        velocity_tracking_width = 0.3

    class asset(Go2TrotCfg.asset):
        # Head_upper and Head_lower are separate bodies from base on the Go2.
        # Run Nov10_21-15-58_ learned to rest its face on the floor at 40 deg
        # pitch because only base contact ended an episode.
        terminate_after_contacts_on = ["base", "Head"]

    class push_robots(Go2TrotCfg.push_robots):
        toggle = False  # no pushes until the box works

    class reward_settings(Go2TrotCfg.reward_settings):
        # Provisional starting value: vertical foot force that counts as contact
        # for touchdown counting, support reward and slip penalty [N]
        # (about 3 % of body weight).
        foot_contact_force_n = 5.0
        # Base height the min_base_height reward saturates at [m]: the reset
        # height. The reference-only run sagged to about 0.20 m with nothing
        # rewarding the body for staying up.
        base_height_target = STANDING_HEIGHT_M + FOOT_RADIUS_M

    class scaling(Go2TrotCfg.scaling):
        # Same scale as base_lin_vel so the critic sees both in the same units.
        reference_body_velocity = BASE_HEIGHT_REF


class Go2BoxStepRunnerCfg(Go2TrotRunnerCfg):
    class actor(Go2TrotRunnerCfg.actor):
        # 52 numbers. No velocity or height commands, no foot contacts.
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
            "step_oscillator_obs",
            "box_phase_obs",
            "step_size",
        ]

    class critic(Go2TrotRunnerCfg.critic):
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
            "step_oscillator_obs",
            "box_phase_obs",
            "step_size",
            "reference_body_velocity",
        ]

        class reward(Go2TrotRunnerCfg.critic.reward):
            # Provisional starting values, modeled on Go2TrotRunnerCfg. None of
            # these weights has been decided by the user; review before training.
            class weights(Go2TrotRunnerCfg.critic.reward.weights):
                # Off: no velocity or height commands in this task.
                tracking_lin_vel = 0.0
                tracking_base_height = 0.0
                trot_support = 0.0
                # Kept from go2trot (same values unless noted).
                tracking_ang_vel = 2.0  # yaw rate target is zero
                lin_vel_z = 0.25  # go2trot has 0.0; the plan keeps this term
                ang_vel_xy = 0.01
                # Raised from go2trot's 1.0: at 1.0, 40 deg of pitch cost only
                # 0.8 and the policy traded it for velocity reward.
                orientation = 3.0
                torques = 5.0e-6
                # Hold the body at the standing height (go2trot has 0.0 here and
                # uses the height command instead; 1.5 is Go2RunnerCfg's value).
                min_base_height = 1.5
                action_rate = 0.25
                action_rate2 = 0.025
                # Pace schedule (same values as trot_support / swing_contact).
                # Raised from 0.625: run Nov13_01-15-36_ kneeled on its shins with
                # its feet down under 10% of the time.
                scheduled_support = 2.0
                # Calf (shin/knee) ground contact, one point per calf body
                # touching (asset.penalize_contacts_on). Was unset, so 0.
                collision = 1.0
                swing_contact = 1.25
                # New box-step terms.
                tracking_body_velocity = 4.0  # as go2trot tracking_lin_vel
                stance_foot_slip = 0.5
                extra_touchdowns = 0.5
                correction_size = 0.1

    class runner(Go2TrotRunnerCfg.runner):
        experiment_name = "go2boxstep"
