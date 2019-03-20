import os, fnmatch, sys
from shutil import copyfile
import numpy as np

task_names = ['Cloth'] #['Blocks', 'Plug', 'Toy_Example]
image_folder = '.' # task_names[0]
data_set1 = ['human1', 'robot1'] #['1', '2', '3', '4', '5', '6']#['bh1', 'bh2', 'bh3', 'bh4']
sample_size1 = [105, 256]
# make all
all_folder = image_folder + '/mixedHR'
if os.path.exists(all_folder) is not True:
    os.mkdir(all_folder)
file_count = 0

dst_size_each = 100
gaps = np.array(sample_size1, dtype=float)/dst_size_each

def addImg(base_folder, t, file_count):
    file_name = base_folder + '/raw_' + str((t + 1)) + '.jpg'
    copyfile(file_name, all_folder + '/raw_' + str((file_count + 1)) + '.jpg')
    file_count += 1
    print("add " + str(file_count))
    return file_count


for i in range(len(data_set1)):
    base_folder = image_folder + '/' + data_set1[i]
    file_count = addImg(base_folder, 0, file_count)
    gap = gaps[i]
    for t in range(dst_size_each-2):
        t_index = int((t+1) * gap)
        file_count = addImg(base_folder, t, file_count)

    file_count = addImg(base_folder, sample_size1[i]-1, file_count)

print(file_count)
