import math
import torch

from go2_deploy.custom_implementation.custom_config import CustomConfig


# Example implementation of the CustomConfig interface for the go2trot task
class Go2TrotBase(CustomConfig):
    task_name = "go2trot"

    # Observation that goes into the network
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "phase_obs",
        "phase_frequency",
    ]

    # Length of observation vector piece (dimensions)
    custom_obs_sizes = {
        "phase_obs": 2,
        "phase_frequency": 1,
    }

    # Default joint angles (rad): FL, FR, RL, RR; hip, thigh, calf in each leg
    default_dof_pos = 4 * [0.0, 0.96, -1.36]

    class DeployScaling:
        base_ang_vel = 0.3
        base_lin_vel = 0.4
        dof_vel = 4 * [2.0, 2.0, 4.0]
        base_height = 0.3
        dof_pos = 4 * [1.0472, 2.53075, 0.94247]
        dof_pos_obs = dof_pos
        dof_pos_target = 4 * [0.2, 0.3, 0.3]
        tau_ff = 4 * [23.7, 23.7, 45.43]
        commands = [3, 1, 3]

    # Initialize any custom buffers your implementation uses (for 1 environment)
    def get_custom_buffers_dict(self):
        test_tensor = torch.zeros(3)

        buffer_dict = {
            "phase_frequency": 2.0,
            "phase": 0.0,
            "test_tensor": test_tensor,
        }

        return buffer_dict

    # Define how your custom buffers should be updated at the control frequency
    def update_custom_buffers(self, main_controller, t):
        main_controller.phase = (
            2 * math.pi * main_controller.phase_frequency * t % (2 * math.pi)
        )

    # Return a dict mapping observation names to functions that generate the observation pieces
    def get_custom_obs_pieces(self):
        obs_pieces = {
            "phase_obs": self._get_obs_phase_obs,
            "phase_frequency": self._get_obs_phase_frequency,
        }
        return obs_pieces

    # Return the phase observation piece of the observation vector
    @staticmethod
    def _get_obs_phase_obs(main_controller, lowstate_msg):
        phase = torch.tensor([main_controller.phase])
        return torch.tensor([torch.sin(phase), torch.cos(phase)])

    # Return the phase frequency observation piece
    @staticmethod
    def _get_obs_phase_frequency(main_controller, lowstate_msg):
        return torch.tensor([main_controller.phase_frequency])
