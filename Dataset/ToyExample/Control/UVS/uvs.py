#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides training and testing module for Bayesian Inference used in visual imitation learning.
=============================================================================
* paper: Robot Visual Imitation Learning using Variational Bayesian Inference
* Author: Jun Jin (jjin5@ualberta.ca), University of Alberta
* TimeMark: Dec 11, 2018
* Status: Dev
* License: check LICENSE in parent folder
=============================================================================
- input: training sample
- output: network module + testing visualizations
- comments: network module will further used in the control module.
"""
import numpy as np
import math
import sys, copy
sys.path.insert(0,'../Robots')

class UVS():

    def __init__(self, robot, broyden_alpha, action_lambda, dim_error, step_normalize_range, estimate_jacobian_random_motion_range):
        self.robot = robot
        self.task_dof = robot.get_robot_dof()
        self.broyden_alpha = broyden_alpha
        self.action_lambda = action_lambda
        self.dim_error = dim_error
        self.jacobian = self.reset_jacobian()
        self.step_normalize_range = step_normalize_range  #[min, max]
        self.random_motion_range = estimate_jacobian_random_motion_range

    def reset_jacobian(self):
        return np.zeros((self.dim_error, self.task_dof))

    def estimate_jacobian_random_motion(self, trial_index):
        """
        step 1, random motion
        :param perturbation_range:
        :return:
        """
        np.random.seed(trial_index)
        random_motion = np.random.uniform(self.random_motion_range[0], self.random_motion_range[1], self.task_dof)
        random_marks = np.random.uniform(0, 1, self.task_dof)
        for i in range(self.task_dof):
            random_marks[i] = 1 if random_marks[i] > 0.5 else -1
        random_motion = random_motion * random_marks
        self.robot.move_step(random_motion)
        return random_motion

    def estimate_jacobian_motion_quality(self, delta_error):
        if np.linalg.norm(delta_error) / math.sqrt(self.task_dof) < 0.01:
            return False # bad quality, not causing error change big enough
        else:
            return True

    def estimate_jacobian(self, delta_joints, delta_errors):
        """
        step 2, input the randome motion and observed delta_error
        :param delta_joints: 2D matrix, n trials * random_motion vectors
        :param delta_errors: 2D matrix, n trials * delta_error vectors
        :return:
        """
        trials = delta_joints.shape[0]
        self.jacobian = self.reset_jacobian()
        for i in range(trials):
            for j in range(self.task_dof):
                self.jacobian[:, j] += delta_errors[i] / delta_joints[i][j]
        self.jacobian = self.jacobian / trials
        print("result jacobian")
        print(self.jacobian)
        return self.jacobian

    def broyden_update_fake(self, delta_error, delta_joint):
        if np.linalg.norm(delta_joint) == 0:
            return False
        print("Broyden Update:")
        print("previous jacobian:")
        print(self.jacobian)
        self.estimate_jacobian(delta_joint.reshape(-1,delta_joint.shape[0]), delta_error.reshape(-1, delta_error.shape[0]))

    def broyden_update(self, delta_error, delta_joint):
        if np.linalg.norm(delta_joint) == 0:
            return False
        dde = (delta_error - np.matmul(self.jacobian, delta_joint)).reshape(self.dim_error, -1)
        update = np.matmul(dde, delta_joint.reshape(-1, self.task_dof)) / np.dot(delta_joint, delta_joint)
        # print("Broyden Update:")
        # print("previous jacobian:")
        # print(self.jacobian)
        self.jacobian = self.jacobian + self.broyden_alpha * update
        # print("updated jacobian")
        # print(self.jacobian)
        return self.jacobian


    def move_step(self, current_error):
        """
        current error can also replaced using generic action
        :param current_error: = target - current, so there is no (-1) here, traditional vs is current-target
        :return:
        """
        step = self.action_lambda * np.matmul(np.transpose(self.jacobian), current_error)
        step_norm = np.linalg.norm(step) # / math.sqrt(self.task_dof)
        if step_norm > self.step_normalize_range[1]:
            print('step norm too large: ' + str(step_norm))
            step = step * self.step_normalize_range[1] / step_norm
        if step_norm < self.step_normalize_range[1]:
            print('step norm too small: ' + str(step_norm))
            step = step * self.step_normalize_range[0] / step_norm
        # step = step * self.step_normalize_range / np.linalg.norm(step)
        self.robot.move_step(step)
        return step





