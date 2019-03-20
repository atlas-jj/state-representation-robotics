import sys
from PIL import Image

image_folder = '../../Dataset/Cloth_prev/human1/'
num = 105

total_output = 20
dest_img_size = 96
dest_offset = 9
total_width = total_output*dest_img_size + (total_output-1)*dest_offset
total_height = 96
gap = num / 20

new_im = Image.new('RGB', (total_width, total_height), color=(255,255,255,0))

def pasteImg(imgIndex, total_index):  #imgIndex start from 1, total Index start from 0
    im = Image.open(image_folder + "raw_" + str(imgIndex)+".jpg")
    im = im.resize((dest_img_size, dest_img_size), Image.ANTIALIAS)
    x_offset = total_index*(dest_img_size + dest_offset)
    new_im.paste(im, (x_offset, 0))

pasteImg(1, 0)

for i in range(total_output-2):
    selected_index = 1 + gap * (i+1)
    pasteImg(int(selected_index), i+1)

pasteImg(num, total_output-1)

new_im.save(image_folder + 'combine.jpg')