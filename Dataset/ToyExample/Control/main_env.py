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
from collections import deque
import os, sys, math, time
sys.path.insert(0,'../../Disentangling/')
sys.path.insert(0,'../../Inference/')
sys.path.insert(0,'../../Lib/')
import  Utils as uts
# sys.path.insert(0,'../../Experiments/Codes/Prepare/')
sys.path.insert(0,'../../Control/High_Level/exp2_toy/')
# sys.path.insert(0,'Robots/')
sys.path.insert(0,'UVS/')
# import inf_models
import inf_utils
import dis_models
import policy_continuous
# import robot_toy_example as robot_model
import uvs
import torch
from colorama import Fore, Back, Style, init as color_init
import matplotlib as mpl
import matplotlib.colors as colors
# from drawnow import drawnow
color_init()

class EnvPlay():
    def __init__(self, robot, dis_model_path, policy_model_path, dim_state, target_state_path, time_varying_marks_path,
                 time_steps=150, image_size=240, dim_rt=1):
        self.magnify_factor = 10
        self.time_varying_marks = np.load(time_varying_marks_path)
        dim_at = self.time_varying_marks.shape[0]
        self.action_ranges = np.ones(dim_at)*0.2
        self.time_steps = time_steps
        self.dim_state = dim_state
        self.dim_at = dim_at
        self.device = torch.device("cuda")
        self.es_jacobian_max_it = 100
        self.dis_model = dis_models.beta_VAE(dim_state, 3).to(self.device)
        self.dis_model.load_weights_from_file(dis_model_path)
        # self.inf_model = inf_models.LfD_Bayesian(dim_state, dim_at, dim_rt).to(self.device)
        # self.inf_model.load_weights_from_file(policy_model_path)
        self.policy_model = policy_continuous.Policy_guided(dim_state, dim_at, self.action_ranges, self.device)
        self.policy_model.load_weights_from_file(policy_model_path)
        self.target_state = np.asarray([-0.29963496, -1.26135778, -0.02019651])  # np.load(target_state_path)
        print('target state')
        print(self.target_state)
        self.image_size = image_size
        self.trajectory = []
        self.robot = robot

        broyden_alpha = 0.5
        action_lambda = 0.1
        dim_error = dim_at
        step_normalize_range = [1.5, 3]
        estimate_jacobian_random_motion_range = [1, 5]
        self.uvs_controller = uvs.UVS(self.robot, broyden_alpha, action_lambda, dim_error, step_normalize_range,
                                      estimate_jacobian_random_motion_range)
        self.inf_helper = inf_utils.inf_model_helper(image_size, self.dis_model, self.time_varying_marks
                                                     , self.device)
        self.initial_state = self.get_current_state()
        self.intermediate_target, _ = self.get_intermediate_target()
        self.thres = 0.1
        self.q_delta_joints = deque(maxlen=5)
        self.q_delta_errors = deque(maxlen=5)

    def sample_all_action_space(self):
        minv=120
        maxv=700
        gap = maxv-minv+1
        dimzs = np.zeros((gap, gap, self.dim_at))
        i_count = 0
        it_x = 0
        for x in range(gap):
            it_y=0
            if x % 1 is 0:
                for y in range(gap):
                    if y % 1 is 0:
                        lean_state = self.get_lean_state_by_robot_position(x+minv,y+minv)
                        dimzs[it_y, it_x, :] = lean_state
                        it_y += 1
                        i_count += 1
                        print(i_count)
                it_x += 1
        return dimzs, it_x


    def main_loop(self, max_step_it=10):
        self.estimate_jacobian()   # initial jacobian estimation using the final target

        self.time_steps = 1000
        for i in range(self.time_steps):
            print('==========  go to step ' +str(i) + '  ========')
            if self.final_task_done():
                break
            input("Press Enter to continue...")

            self.intermediate_target, generic_action = self.get_intermediate_target()
            errors = []
            for i in range(max_step_it):
                if self.intermediate_task_done(self.intermediate_target):
                    break
                # r_error, _ = self.get_error(self.intermediate_target)
                r_error, _ = self.get_error(self.target_state)
                errors.append(np.linalg.norm(r_error))
                r_motion, delta_error = self.execute_step(r_error)
                # if self.error_reverse(errors):
                #     print('error reverse')
                #     self.brute_force_jacobian_update()
                error_std = self.last_error_std(errors)
                print('std: '+str(error_std))
                if np.linalg.norm(self.uvs_controller.jacobian) < 0.01 or error_std < 0.01 or self.error_reverse(errors):
                    print('jacobian too small or error std too small')
                    # robot trapped, move to a random motion
                    errors.clear()
                    self.uvs_controller.estimate_jacobian_random_motion(999)
                    self.estimate_jacobian()
                else:
                    self.q_delta_joints.append(r_motion)
                    self.q_delta_errors.append(delta_error)

                # input("Press Enter to continue...")

    def error_reverse(self, errors):
        length = len(errors)
        n = 5
        if length < (n+1):
            return False
        else:
            for i in range(n):
                if errors[length-1-i] < errors[length-2-i]: # normal
                    return False
            return True

    def last_error_std(self, errors, last_n=4):
        length = len(errors)
        if length < last_n:
            return 999
        else:
            # print(errors[length-last_n:length])
            return np.std(errors[length-last_n:length])

    def print_latent_space(self):
        current_state = self.get_current_state()
        for i in range(self.dim_at):
            self.print_latent_space_single(self.initial_state[self.time_varying_marks[i]], current_state[
                self.time_varying_marks[i]], self.intermediate_target[i], self.target_state[i])

    def print_latent_space_single(self, initial_v, current_v, intermediate_t, final_t):
        print(Fore.GREEN + uts.d2s(initial_v)+'---'+uts.d2s(current_v)+'->->['+uts.d2s(intermediate_t-current_v)+']->->'+
              uts.d2s(intermediate_t)+'->->->->->['+uts.d2s(final_t-current_v)+']->->->->->'+uts.d2s(final_t) + Style.RESET_ALL)

    def estimate_jacobian(self, trials=5):
        print('estimate jacobian')
        # self.set_intermediate_target(intermediate_target)  ####
        current_joints = self.robot.current_joints()  # record its original position
        delta_joints = []
        delta_errors = []

        prev_state = self.get_current_state_lean()
        print('initial state')
        print(prev_state)
        for i in range(trials):
            # print('prev_error')
            # print(prev_error)
            # input("Press Enter to continue...")
            print('estimate jacobian, now move robot')
            # input("Press Enter to continue...")
            r_motion = self.uvs_controller.estimate_jacobian_random_motion(i)  # move robo
            print(r_motion)
            r_state = self.get_current_state_lean()
            print('new state')
            print(r_state)

            delta_error = (r_state - prev_state) * self.magnify_factor
            print('delta error')
            print(delta_error)
            if self.uvs_controller.estimate_jacobian_motion_quality(delta_error):
                delta_joints.append(r_motion)
                delta_errors.append(delta_error)
                self.q_delta_joints.append(r_motion)
                self.q_delta_errors.append(delta_error)
            else:
                print('bad quality jacobian')
            # move back to current joints
            print('estimate jacobian, now back to origin')
            # input("Press Enter to continue...")
            self.robot.move_to(current_joints)  # back to its original position
            print('robot move back to origin')

        self.estimate_new_jacobian(delta_joints, delta_errors)

    def brute_force_jacobian_update(self):
        if len(self.q_delta_joints) > 2:
            print('brute_force_jacobian_update')
            self.estimate_new_jacobian(self.q_delta_joints, self.q_delta_errors)

    def estimate_new_jacobian(self, delta_joints, delta_errors):
        delta_joints = np.asarray(delta_joints)
        delta_errors = np.asarray(delta_errors)
        self.uvs_controller.estimate_jacobian(delta_joints, delta_errors)

    def estimate_jacobian_with_target(self, intermediate_target, trials=3):
        print('estimate jacobian')
        # self.set_intermediate_target(intermediate_target)  ####
        current_joints = self.robot.current_joints()  # record its original position
        delta_joints = []
        delta_errors = []

        prev_error, current_state = self.get_error(intermediate_target)
        print('current state')
        print(current_state)
        for i in range(trials):
            # print('prev_error')
            # print(prev_error)
            # input("Press Enter to continue...")
            print('estimate jacobian, now move robot')
            input("Press Enter to continue...")
            r_motion = self.uvs_controller.estimate_jacobian_random_motion(i)  # move robo
            print(r_motion)
            r_error, r_state = self.get_error(intermediate_target)
            print('new state')
            print(r_state)

            delta_error = r_error - prev_error
            print('delta error')
            print(delta_error)
            if self.uvs_controller.estimate_jacobian_motion_quality(delta_error):
                delta_joints.append(r_motion)
                delta_errors.append(delta_error)
            else:
                print('bad quality jacobian')
            # move back to current joints
            print('estimate jacobian, now back to origin')
            input("Press Enter to continue...")
            self.robot.move_to(current_joints)  # back to its original position
            print('robot move back to origin')

        delta_joints = np.asarray(delta_joints)
        delta_errors = np.asarray(delta_errors)
        self.uvs_controller.estimate_jacobian(delta_joints, delta_errors)

    def execute_step(self, generic_action):
        it_count = 0
        # self.set_intermediate_target(intermediate_target)  # ####
        self.trajectory.append(self.robot.get_recording_state())
        prev_state = self.get_current_state_lean()

        print('prev_state')
        print(prev_state)
        r_motion = self.uvs_controller.move_step(generic_action)
        print('step:')
        print(r_motion)
        self.print_latent_space()
        current_joints = self.robot.current_joints()
        self.trajectory.append(self.robot.get_recording_state())
        # print('current state x: ' + str(current_joints[0]) + '  y: ' + str(current_joints[1]))
        current_state = self.get_current_state_lean()
        print('current_state')
        print(current_state)
        delta_error = (current_state - prev_state) * self.magnify_factor
        print('delta error')
        print(delta_error)
        print('delta_joints')
        print(r_motion)
        # broyden update
        self.uvs_controller.broyden_update(delta_error, r_motion)
        return r_motion, delta_error


    def execute_step_deprecated(self, intermediate_target, max_it=100):
        it_count = 0
        # self.set_intermediate_target(intermediate_target)  # ####
        self.trajectory.append(self.robot.get_recording_state())
        prev_error, prev_state = self.get_error(intermediate_target)
        while it_count < max_it and self.intermediate_task_done(intermediate_target) is not True:
            self.print_latent_space()
            print('prev_state')
            print(prev_state)
            r_motion = self.uvs_controller.move_step(prev_error)
            print('step:')
            print(r_motion)
            current_joints = self.robot.current_joints()
            self.trajectory.append(self.robot.get_recording_state())
            # print('current state x: ' + str(current_joints[0]) + '  y: ' + str(current_joints[1]))
            r_error, current_state = self.get_error(intermediate_target)
            print('current_state')
            print(current_state)
            delta_error = (current_state - prev_state) * self.magnify_factor
            print('delta error')
            print(delta_error)
            print('delta_joints')
            print(r_motion)
            # broyden update
            self.uvs_controller.broyden_update(delta_error, r_motion)

            prev_error = r_error
            input("Press Enter to continue...")
            if np.linalg.norm(self.uvs_controller.jacobian) < 0.01:
                print('jacobian too small')
                self.estimate_jacobian(intermediate_target)
            it_count += 1
        print('step done')

    def execute_step_with_target(self, intermediate_target, max_it=100):
        it_count = 0
        # self.set_intermediate_target(intermediate_target)  # ####
        self.trajectory.append(self.robot.get_recording_state())
        prev_error, prev_state = self.get_error(intermediate_target)
        while it_count < max_it and self.intermediate_task_done(intermediate_target) is not True:
            self.print_latent_space()
            print('prev_error')
            print(prev_error)
            r_motion = self.uvs_controller.move_step(prev_error)
            print('step:')
            print(r_motion)
            current_joints = self.robot.current_joints()
            self.trajectory.append(self.robot.get_recording_state())
            print('current state x: ' + str(current_joints[0]) + '  y: ' + str(current_joints[1]))
            r_error, _ = self.get_error(intermediate_target)
            print('current error')
            print(r_error)
            delta_error = r_error - prev_error
            print('delta error')
            print(delta_error)
            print('delta_joints')
            print(r_motion)
            # broyden update
            self.uvs_controller.broyden_update(delta_error, r_motion)

            prev_error = r_error
            input("Press Enter to continue...")
            if np.linalg.norm(self.uvs_controller.jacobian) < 0.01:
                print('jacobian too small')
                self.estimate_jacobian(intermediate_target)
            it_count += 1
        print('task done')


    def final_task_done(self):
        err, _ = self.get_error(self.target_state)
        final_error = np.linalg.norm(err)
        print(Fore.YELLOW + Back.BLUE + 'final error  ' + str(final_error) + Style.RESET_ALL)
        if final_error < self.thres*self.magnify_factor:
            print('final done')
            return True
        else:
            return False

    def intermediate_task_done(self, intermediate_target):
        err, _ = self.get_error(intermediate_target)
        intermediate_error = np.linalg.norm(err)
        print(Fore.CYAN + 'intermediate error ' + str(intermediate_error) + Style.RESET_ALL)
        if intermediate_error < self.thres*self.magnify_factor:
            print('intermediate done')
            return True
        else:
            return False

    def lean_state(self, St):
        lean_st = np.zeros(self.dim_at)
        for i in range(self.dim_at):
            lean_st[i] = St[self.time_varying_marks[i]]
        return lean_st

    def generic_system_dynamics(self, St, at):
        next_state = np.random.normal(0, 1, self.dim_state)
        intermediate_target = np.zeros(self.dim_at)
        for i in range(self.dim_at):
            next_state[self.time_varying_marks[i]] = St[self.time_varying_marks[i]] + at[i]
            intermediate_target[i] = St[self.time_varying_marks[i]] + at[i]
        return next_state, intermediate_target

    def get_error(self, intermediate_target):
        current_image = self.robot.get_current_image()
        current_state = self.inf_helper.get_latent_state(current_image)
        current_lean_state = self.lean_state(current_state)
        error = intermediate_target - current_lean_state
        return error*self.magnify_factor, current_lean_state

    def get_current_state(self):
        current_image = self.robot.get_current_image()
        current_state = self.inf_helper.get_latent_state(current_image)
        return current_state

    def get_lean_state_by_robot_position(self, x, y):
        self.robot.move_to(np.asarray([x,y]))
        return self.get_current_state_lean()

    def get_current_state_lean(self):
        return self.lean_state(self.get_current_state())

    def get_current_at_star(self):
        current_state = self.get_current_state()
        at_star = self.target_state - self.lean_state(current_state)
        return at_star  # dont use magnify factor when feed to policy network

    def get_generic_action(self):
        current_state = self.get_current_state()
        at_star = self.target_state - self.lean_state(current_state)
        generic_action, _ = self.policy_model.sample_action(current_state, at_star)  # dont use magnify factor when feed to policy network
        return generic_action[0]

    def get_intermediate_target(self):
        current_state = self.get_current_state()
        generic_action = self.get_generic_action()
        next_state, intermediate_target = self.generic_system_dynamics(current_state, generic_action)  # system dynamics, all use magnify factor
        return intermediate_target, generic_action

    def set_intermediate_target_fake(self, intermediate_target):
        current_joints = self.robot.current_joints()
        self.robot.set_intermediate_target_fake(current_joints[0] + intermediate_target[0], current_joints[1]
                                           + intermediate_target[1])



