import os, sys, math, time
sys.path.insert(0,'Robots/')
sys.path.insert(0,'../../Experiments/Codes/Prepare/')
import robot_toy_example as robot_model
import main_env
import numpy as np

dis_model_path = '../../Disentangling/params/Toy_Example_all_dimZ_100alpha_0.2'
policy_model_path = '../../Control/High_Level/exp2_toy/params/toy_example_policy_lr3_v2_good'
time_varying_marks_path = '../../Disentangling/results/toy_example_ZtMarks.npy'
target_state_np_path = '../../Inference/results/Toy_Example/Toy_Example_DimZ100_0.2/target.npy'
dim_state = 100
max_time_steps = 150
robot = robot_model.toy_blocks_robot()

myEnv = main_env.EnvPlay(robot, dis_model_path, policy_model_path, dim_state, target_state_np_path, time_varying_marks_path,
                         max_time_steps, image_size=240, dim_rt=1)

# myEnv.main_loop()
dimzs , cols_count= myEnv.sample_all_action_space()
np.save('toy_example_all_sampling_gap4', dimzs)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
color_trajectory = plt.pcolor(dimzs[0:cols_count,0:cols_count,0], cmap='RdBu')
plt.colorbar(color_trajectory)
