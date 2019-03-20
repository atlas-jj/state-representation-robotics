import os, fnmatch, sys
import natsort
folder = 'final_states'

pattern = "*.jpg"
# delete all jpg FILES

listOfFiles = os.listdir('./'+folder)
sample_size = 0
img_names = []
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        sample_size += 1
        img_names.append(entry)

img_names = natsort.natsorted(img_names)

for i in range(len(img_names)):
    os.rename('./' + folder + '/' + img_names[i], './' + folder + '/raw_' + str(i+1) + '.jpg')
    print('rename ' + 'raw_' + str(i+1))



