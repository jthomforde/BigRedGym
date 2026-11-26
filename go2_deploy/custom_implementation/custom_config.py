"""Interface every custom deployment implementation must extend.

Put a file in go2_deploy/custom_implementation/ that defines one subclass of
CustomConfig with a unique `task_name` (see go2trot_base.py for an example).
load_custom_impl(task_name) finds it by that name.

Like a Java interface, leaving something out is an error: a missing class
attribute raises TypeError when the subclass is defined, and a missing method
raises TypeError when it is instantiated.
"""

from abc import ABC, abstractmethod


class CustomConfig(ABC):
    # Class attributes every subclass must define, with their expected types.
    REQUIRED_ATTRIBUTES = {
        # Task name (e.g. "go2trot"): the policy is loaded from the latest run
        # (or --run_name) in logs/<task_name>/, whose saved gym/envs must
        # declare this task. Also how load_custom_impl() finds this class.
        "task_name": str,
        # Observation names, in the order they go into the network
        "obs_vector": list,
        # Dimension of each custom (non built-in) observation piece
        "custom_obs_sizes": dict,
        # Class whose attributes are the scale for each observation piece
        "DeployScaling": type,
        # Static default posture (rad) that actions are offsets from: 12
        # joint angles in FL, FR, RL, RR order, hip, thigh, calf within each
        # leg. Must match the default joint angles the policy was trained with.
        "default_dof_pos": list,
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [name for name in cls.REQUIRED_ATTRIBUTES if not hasattr(cls, name)]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define class attribute(s): {', '.join(missing)}"
            )
        for name, expected in cls.REQUIRED_ATTRIBUTES.items():
            if not isinstance(getattr(cls, name), expected):
                raise TypeError(
                    f"{cls.__name__}.{name} must be a {expected.__name__}, "
                    f"got {type(getattr(cls, name)).__name__}"
                )

    # Initial values of the custom buffers your implementation uses (for 1
    # environment). Each key becomes an attribute on the MainController.
    @abstractmethod
    def get_custom_buffers_dict(self) -> dict: ...

    # Update the custom buffers on main_controller; called every control step
    # with t, the time in seconds.
    @abstractmethod
    def update_custom_buffers(self, main_controller, t) -> None: ...

    # Map each name in custom_obs_sizes to a function
    # f(main_controller, lowstate_msg) -> torch.Tensor of that size.
    @abstractmethod
    def get_custom_obs_pieces(self) -> dict: ...
