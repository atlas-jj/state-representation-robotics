import numpy as np
import math
import sys, copy
sys.path.insert(0,'../Robots')
import robot_toy_example as robot_moel
import uvs as uvss
import time


robot = robot_moel.toy_blocks_robot()
uvs = uvss.UVS(robot, 0.5, 0.1, 2, 5)
trajectory = []

def set_intermediate_target(dx, dy):
    current_joints = robot.current_joints()
    robot.set_intermediate_target_fake(current_joints[0] + dx, current_joints[1] + dy)

def estimate_jacobian(trails=10):
    current_joints = robot.current_joints()
    delta_joints = []
    delta_errors = []
    trials = 10
    prev_error = robot.get_fake_error()
    for i in range(trials):

        input("Press Enter to continue...")
        r_motion = uvs.estimate_jacobian_random_motion(5, i)

        r_error = robot.get_fake_error()
        delta_error = r_error - prev_error

        if uvs.estimate_jacobian_motion_quality(delta_error):
            delta_joints.append(r_motion)
            delta_errors.append(delta_error)
        # move back to current joints
        input("Press Enter to continue...")
        robot.move_to(current_joints)
        print('back to origin')
    delta_joints = np.asarray(delta_joints)
    delta_errors = np.asarray(delta_errors)
    uvs.estimate_jacobian(delta_joints, delta_errors)

max_it = 100
def go_loop():
    it_count = 0
    trajectory.append(robot.get_recording_state())
    while it_count < max_it and robot.task_done() is not True:
        prev_error = robot.get_fake_error()
        r_motion = uvs.move_step(robot.get_fake_error())
        print('step:')
        print(r_motion)
        current_joints = robot.current_joints()
        trajectory.append(robot.get_recording_state())
        print('current state x: ' + str(current_joints[0]) + '  y: ' + str(current_joints[1]))
        r_error = robot.get_fake_error()
        delta_error = r_error - prev_error
        print('delta error')
        print(delta_error)
        print('delta_joints')
        print(r_motion)
        # broyden update
        uvs.broyden_update(delta_error, r_motion)
        input("Press Enter to continue...")
        it_count += 1
    print('task done')

# current_error = robot.get_fake_error()