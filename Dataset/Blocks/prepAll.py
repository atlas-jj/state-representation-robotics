from PIL import Image, ImageDraw, ImageChops, ImageFilter
import os, fnmatch, sys

task_name = "BlocksV2"

src = "./blocks_robot_2.avi"
dest = "robot2"

# mask_top_left = [711,47]
# mask_bottom_right = [1493,829]

mask_top_left = [700,0]
mask_bottom_right = [1780,1080]
# mask_top_left = [460,0]
# mask_bottom_right = [1390,930]

pattern = "*.jpg"
# delete all jpg FILES
os.system("mkdir " + dest)
os.system("rm -rf -d ./"+dest+"/*.jpg")
os.system("ffmpeg -i "+src+" ./"+dest+"/%d.jpg")
input_size = 240 # input raw image size, to the network

listOfFiles = os.listdir('./'+dest)
sample_size = 0
img_names = []
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        sample_size += 1
        img_names.append(entry)

print("1. please define the final image size you want to use:")
sample_size = int(input("Enter a number: "))

def process_img(src, crop1, crop2, resize_w):
    cropped = src.crop((crop1[0], crop1[1], crop2[0], crop2[1]))
    #cropped.show()
    cropped = cropped.filter(ImageFilter.GaussianBlur(2))
    cropped = cropped.resize((resize_w, resize_w))
    #cropped.show()
    #input("Press Enter to continue...")
    return cropped

print("2. please define raw sample gap, default 2:")
raw_gap = 2
raw_gap = int(input("Enter a number: "))
raw_sample_size = 0
for i in range(sample_size):
    img = Image.open('./'+dest+'/' + str(i+1) + '.jpg')#
    # idx = int(img_names[i].split('.')[0].split('_')[1])
    print('index:' + str(i))
    if ((i+1) % raw_gap ==0):
        raw_sample_size += 1
        cropped = process_img(img, mask_top_left, mask_bottom_right, 240)
        cropped.save('./'+dest+'/raw_'+str(int(i/raw_gap+1))+'.jpg')

# now delete all frame images
for i in range (len(img_names)):
    os.remove('./'+dest + '/' + img_names[i])
print("frame images deleted!")

