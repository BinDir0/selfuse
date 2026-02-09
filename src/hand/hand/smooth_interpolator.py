#!/usr/bin/env python3
"""
Smooth Joint Interpolator using Ruckig library
Ported from mj-controller/haptic_hand_control/Ruckig_Interpolator.py
Kept consistent with the teleoperation codebase
"""

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig


class SmoothJointInterpolator:
    """
    Ruckig-based smooth joint interpolator with time synchronization
    and multi-DOF support.
    """

    def __init__(self, dof: int, step: float, alpha: float):
        """
        Args:
            dof: Number of degrees of freedom.
            step: Control period in seconds (1.0 / output_freq).
            alpha: Low-pass filter coefficient for input smoothing.
        """
        self.dof = dof
        self.step = step
        self.alpha = alpha

        # Ruckig instance
        self.ruckig = Ruckig(self.dof, self.step)
        self.input_param = InputParameter(self.dof)
        self.output_param = OutputParameter(self.dof)

        # Default kinematic limits
        self.input_param.max_velocity = [3.0] * self.dof
        self.input_param.max_acceleration = [6.0] * self.dof
        self.input_param.max_jerk = [12.0] * self.dof

        self.result = Result.Working
        self.param_initialized = False

    def set_kinematic_limits(
        self, max_velocity=None, max_acceleration=None, max_jerk=None
    ):
        """Set kinematic limits (max velocity, acceleration, jerk)."""
        if max_velocity is not None:
            self.input_param.max_velocity = (
                [max_velocity] * self.dof
                if isinstance(max_velocity, (int, float))
                else max_velocity
            )
        if max_acceleration is not None:
            self.input_param.max_acceleration = (
                [max_acceleration] * self.dof
                if isinstance(max_acceleration, (int, float))
                else max_acceleration
            )
        if max_jerk is not None:
            self.input_param.max_jerk = (
                [max_jerk] * self.dof
                if isinstance(max_jerk, (int, float))
                else max_jerk
            )

    def set_input_param(
        self, current_position, current_velocity=None, current_acceleration=None
    ):
        """Set current state."""
        self.input_param.current_position = current_position
        if current_velocity is not None:
            self.input_param.current_velocity = current_velocity
        else:
            self.input_param.current_velocity = np.zeros(self.dof)
        if current_acceleration is not None:
            self.input_param.current_acceleration = current_acceleration
        else:
            self.input_param.current_acceleration = np.zeros(self.dof)
        self.param_initialized = True

    @property
    def is_initialized(self):
        return self.param_initialized

    def update(self, target_pos, target_vel=None, target_acc=None):
        """
        Update target position and compute next interpolated state.

        Returns:
            Tuple of (position, velocity, acceleration, is_working)
        """
        if not self.is_initialized:
            raise ValueError("Interpolator is not initialized")

        self.input_param.target_position = target_pos
        self.input_param.target_velocity = (
            np.zeros(self.dof) if target_vel is None else target_vel
        )
        self.input_param.target_acceleration = (
            np.zeros(self.dof) if target_acc is None else target_acc
        )

        self.result = self.ruckig.update(self.input_param, self.output_param)
        working = Result.Working == self.result
        self.output_param.pass_to_input(self.input_param)

        pos = np.array(self.output_param.new_position)
        vel = np.array(self.output_param.new_velocity)
        acc = np.array(self.output_param.new_acceleration)

        return pos, vel, acc, working

    def close(self):
        """Release resources."""
        self.param_initialized = False
