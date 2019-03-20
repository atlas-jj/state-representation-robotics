import os, fnmatch, sys
from shutil import copyfile


task_names = ['Cloth_prev'] #['Blocks', 'Plug', 'Toy_Example]
image_folder = '../../Dataset/' + task_names[0]
data_set1 = ['human1', 'human2']  #['1', '2', '3', '4', '5', '6']#['bh1', 'bh2', 'bh3', 'bh4']
sample_size1 = [105, 111]
# make all
all_folder = image_folder + '/human12'
if os.path.exists(all_folder) is not True:
    os.mkdir(all_folder)
file_count = 0
for i in range(len(data_set1)):
    base_folder = image_folder + '/' + data_set1[i]
    for t in range(sample_size1[i]):
        file_name = base_folder + '/raw_' + str((t+1)) + '.jpg'
        copyfile(file_name, all_folder + '/raw_' + str((file_count+1)) + '.jpg')
        file_count += 1

print(file_count)
