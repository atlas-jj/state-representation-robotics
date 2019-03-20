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
import sys, copy
sys.path.insert(0,'../../../Experiments/Codes/Prepare/')
import toy_model
from tkinter import *

class toy_blocks_robot():

    def __init__(self, active_joints_mark=[1,2]):
        """

        :param active_joints_mark: [1,2], or [1], or [2]
        """
        self.active_joints_mark = active_joints_mark
        self.task_dof = len(active_joints_mark)
        self.toy_robot = toy_model.PlayGroundBlocks()
        self.toy_robot.mode = 2  # for uvs testing

    def get_robot_dof(self):
        return self.task_dof

    def get_active_joints(self, full_joints):
        active_joints = np.zeros(self.task_dof)
        for i in range(self.task_dof):
            active_joints[i] = full_joints[self.active_joints_mark[i] -1]
        return active_joints

    def get_fake_error(self):
        c_joints = self.current_joints()
        return np.asarray([self.toy_robot.target_x - c_joints[0], self.toy_robot.target_y - c_joints[1]])

    def set_intermediate_target_fake(self, x, y):
        self.toy_robot.target_x = x
        self.toy_robot.target_y = y
        currentxy = self.toy_robot.get_current_x_y()
        self.toy_robot.move_step(currentxy[0], currentxy[1])

    def task_done_fake(self):
        return self.toy_robot.task_done()

    def current_joints(self):
        """
        only returns current joints represented in active joints mark
        :return:
        """
        all_joints = self.toy_robot.get_current_x_y()

        return self.get_active_joints(all_joints)

    def set_new_joints(self, delta_joints):
        """

        :param delta_joints: only contains delta value for active joints
        :return:
        """
        all_joints = self.toy_robot.get_current_x_y()
        for i in range(self.task_dof):
            all_joints[self.active_joints_mark[i] - 1] += delta_joints[i]

        return all_joints

    def set_active_joints(self, active_joints):
        all_joints = self.toy_robot.get_current_x_y()
        for i in range(self.task_dof):
            all_joints[self.active_joints_mark[i] - 1] = active_joints[i]
        return all_joints

    def get_recording_state(self):
        return np.asarray([self.toy_robot.current_x, self.toy_robot.current_y])

    def move_step(self, delta_joints):
        new_joints = self.set_new_joints(delta_joints)
        self.toy_robot.move_step(new_joints[0], new_joints[1])

    def move_to(self, active_joints):
        new_joints = self.set_active_joints(active_joints)
        self.toy_robot.move_step(new_joints[0], new_joints[1])

    def get_current_image(self):
        """
        return a PIL image object.
        :return:
        """
        return self.toy_robot.get_image(self.toy_robot.current_x, self.toy_robot.current_y)


