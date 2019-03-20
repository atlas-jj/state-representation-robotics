from tkinter import *
import time
from tkinter import messagebox, filedialog
import numpy as np
import os.path
from PIL import Image, ImageDraw
import shutil


class PlayGroundBlocks(Frame):
    def __init__(self, c_width=800, c_height=800, target_x=700, target_y=700, rect_width=100, save_file='my_way', parent=None):
        Frame.__init__(self, parent)
        self.pack()
        self.master.title("A simple paint program")
        self.is_drawing = False
        self.thres = 5
        self.traj_num = StringVar()
        self.traj_num.set("Recorded Trajectories: 0")
        self.make_widgets()
        self.canvas_width = c_width
        self.canvas_height = c_height
        self.target_x = target_x
        self.target_y = target_y
        self.rect_width = rect_width
        self.save_file = save_file
        self.current_x = 20
        self.current_y = 20
        self.mouse_start_x = 0
        self.mouse_start_y = 0
        self.current_traj = []
        self.trajectories = []
        self.mode = 0  # 0: drawing, 1: display trajectories, 2: for uvs testing

        canvas = Canvas(width=self.canvas_width, height=self.canvas_height, bg='black')
        canvas.bind('<ButtonPress-1>', self.onStart)
        canvas.bind('<B1-Motion>',     self.onGrow)
        # canvas.bind('<Double-1>',      self.onClear)
        # canvas.bind('<ButtonPress-3>', self.onMove)
        self.canvas = canvas
        self.canvas.pack()
        # initialize drawing
        self.random_init_block()

    def get_current_x_y(self):
        return np.asarray([self.current_x + self.rect_width, self.current_y + self.rect_width])

    def make_widgets(self):
        Label(self, textvariable=self.traj_num).pack()
        Label(self, text='Click on the blue square, then drag it to the target diamond.').pack()
        Button(self, text='Start Again', command=self.random_init_block).pack(side=LEFT)
        Button(self, text='Clear All', command=self.restart).pack(side=LEFT)
        Button(self, text='Load Trajectories', command=self.load_trajs).pack(side=LEFT)

    def random_init_block(self):
        # draw initial blocks
        self.mode = 0
        start_x = np.random.randint(10, self.canvas_width/3 - self.rect_width)
        start_y = np.random.randint(10, self.canvas_height/3 - self.rect_width)
        self.current_traj = []
        self.refresh(start_x, start_y)

    def move_step(self, endx, endy):
        self.refresh(endx - self.rect_width, endy - self.rect_width)

    def task_done(self):
        self.thres = 0.2
        if abs(self.current_x+self.rect_width-self.target_x) < self.thres and abs(self.current_y+self.rect_width
                                                                                  -self.target_y) < self.thres:
            return True
        else:
            return False

    def refresh(self, start_x, start_y):
        if self.mode is 1:
            return
        if self.mode is 0 and start_x + self.rect_width > self.target_x and start_y + self.rect_width > self.target_y:
            return
        self.current_x = start_x
        self.current_y = start_y
        self.canvas.delete("all")
        # draw all previous trajs
        self.draw_all_trajs(self.trajectories)

        self.canvas.create_rectangle(start_x, start_y, start_x + self.rect_width, start_y + self.rect_width, fill='blue')
        self.canvas.create_rectangle(start_x - 10, start_y - 10, start_x + 10, start_y + 10, fill='yellow')
        self.canvas.create_oval(start_x + self.rect_width - 10, start_y - 10, start_x + self.rect_width + 15,
                                start_y + 15, fill='white')
        self.canvas.create_polygon(start_x, start_y + self.rect_width, start_x - 10, start_y + self.rect_width + 15, start_x + 10,
                                   start_y + self.rect_width + 15, fill='green')
        self.canvas.create_polygon(start_x + self.rect_width, start_y + self.rect_width - 20, start_x + self.rect_width + 10,
                                   start_y + self.rect_width,
                                   start_x + self.rect_width, start_y + self.rect_width + 20, start_x + self.rect_width - 10,
                                   start_y + self.rect_width
                                   , fill='red')
        # draw the target
        self.canvas.create_polygon(self.target_x, self.target_y - 20, self.target_x + 10, self.target_y,
                                   self.target_x, self.target_y + 20, self.target_x - 10, self.target_y, fill='red')

        # draw the trajectory
        self.draw_traj_point(start_x, start_y)
        # draw current trajectory
        for i in range(len(self.current_traj)):
            self.draw_traj_point(self.current_traj[i][0], self.current_traj[i][1])

        # detect if reaches the target
        if self.mode is 0 and abs(start_x+self.rect_width-self.target_x) < self.thres and \
                abs(start_y+self.rect_width-self.target_y) < self.thres:
            messagebox.showinfo("Well done", "Good job! You did it!")
            self.record_another_one()

        # time.sleep(0.25)

    def get_image(self, start_x, start_y, save_file_name='default'):
        # if start_x + self.rect_width > self.target_x and start_y + self.rect_width > self.target_y:
        #     return
        white=(255, 255, 255)
        black = (0, 0, 0)
        blue = (0,0,255)
        red = (255, 0, 0)
        green = (0,255,0)
        yellow = (255,255,0)

        image1 = Image.new("RGB", (self.canvas_width, self.canvas_height), black)
        draw = ImageDraw.Draw(image1)
        draw.rectangle([start_x, start_y, start_x + self.rect_width, start_y + self.rect_width], fill=blue)
        draw.rectangle([start_x - 10, start_y - 10, start_x + 10, start_y + 10], fill=yellow)
        draw.pieslice([start_x + self.rect_width - 10, start_y - 10, start_x + self.rect_width + 15,
                                start_y + 15], 0, 360, fill=white)
        draw.polygon([start_x, start_y + self.rect_width, start_x - 10, start_y + self.rect_width + 15, start_x + 10,
                                   start_y + self.rect_width + 15], fill=green)
        draw.polygon([start_x + self.rect_width, start_y + self.rect_width - 20, start_x + self.rect_width + 10,
                                   start_y + self.rect_width,
                                   start_x + self.rect_width, start_y + self.rect_width + 20, start_x + self.rect_width - 10,
                                   start_y + self.rect_width], fill=red)

        # draw the target
        draw.polygon([self.target_x, self.target_y - 20, self.target_x + 10, self.target_y,
                                   self.target_x, self.target_y + 20, self.target_x - 10, self.target_y], fill=red)
        draw.pieslice([start_x + self.rect_width - 2, start_y + self.rect_width - 2, start_x + self.rect_width + 2,
                  start_y+ self.rect_width + 2], 0,360,fill=blue)

        if save_file_name is not 'default':
           image1.save(save_file_name+'.jpg')
        return image1


    def draw_all_trajs(self, all_trajs):
        # draw the trajectory
        for i in range(len(all_trajs)):
            c = all_trajs[i]
            for j in range(len(c)):
                x = c[j][0]
                y = c[j][1]
                self.draw_traj_point(x, y)


    def draw_traj_point(self, x, y):
        self.canvas.create_oval(x + self.rect_width - 2, y + self.rect_width - 2, x + self.rect_width + 2,
                                y + self.rect_width + 2, fill='cyan')

    def record_another_one(self):
        # save to traj List
        self.trajectories.append(self.current_traj)
        self.update_traj_num()
        # save current traj to file
        toSave = []
        for i in range(len(self.trajectories)):
            c = self.trajectories[i]
            this_traj = np.zeros((len(c), 8))
            for j in range(len(c)):
                this_traj[j, :] = self.get_corner_points(c[j][0], c[j][1])
            toSave.append(this_traj)

        np.save(self.save_file, toSave)
        self.random_init_block()
        self.update_traj_num()

    def get_corner_points(self, start_x, start_y):
        c = np.zeros((1, 8))
        c[0,0] = start_x
        c[0,1] = start_y
        c[0,2] = start_x + self.rect_width
        c[0,3] = start_y
        c[0,4] = start_x + self.rect_width
        c[0,5] = start_y + self.rect_width
        c[0,6] = start_x
        c[0,7] = start_y + self.rect_width
        return c

    # clear all
    def restart(self):
        self.current_traj = []
        self.trajectories[:] = []
        self.update_traj_num()
        self.random_init_block()

    def update_traj_num(self):
        self.traj_num.set("Recorded Trajectories: " + str(len(self.trajectories)))


    def onStart(self, event):
        # print("onstart...")
        self.mouse_start_x = event.x
        self.mouse_start_y = event.y

    def onGrow(self, event):
        # print("on Grow move" + str(event.x) + ";" + str(event.y))
        diffX, diffY = (event.x - self.mouse_start_x), (event.y - self.mouse_start_y)
        self.mouse_start_x = event.x
        self.mouse_start_y = event.y
        self.refresh(self.current_x + diffX, self.current_y + diffY)
        # set to current traj, only save the startX and startY
        self.current_traj.append([self.current_x + diffX, self.current_y + diffY])

    def quit(self):
        Frame.quit(self)

    def onClear(self, event):
        self.random_init_block()

    def valid(self, index):
        not_included = [4,5]
        for i in range(len(not_included)):
            if index is not_included[i]:
                return False
        return True

    def load_trajs(self):
        filename = filedialog.askopenfilename()
        print(filename)
        if os.path.isfile(filename):
            trajs = np.load(str(filename))
            new_trajs = []
            # draw all previous trajs
            for i in range(len(trajs)):
                if self.valid(i):
                    c = []
                    for j in range(trajs[i].shape[0]):
                        c.append([trajs[i][j, 0], trajs[i][j, 1]])
                    new_trajs.append(c)
            print("selected trajectories: " + str(len(new_trajs)))
            self.mode = 1  # 1: draw trajectories
            self.canvas.delete("all")
            self.draw_all_trajs(new_trajs)
            self.traj_num.set("Recorded Trajectories: " + str(len(new_trajs)))



class GenerateImages:
    def __init__(self, trajs_path, model, traj_name):
        self.model = model
        trajs = np.load(str(trajs_path))
        self.traj_name = traj_name
        self.all_trajs = []
        for i in range(len(trajs)):
            c = []
            for j in range(trajs[i].shape[0]):
                c.append([trajs[i][j, 0], trajs[i][j, 1]])
            self.all_trajs.append(c)

    def save_img(self, c, indexj, saveFolder, inner_index):
        x = c[indexj][0]
        y = c[indexj][1]
        img = self.model.get_image(x, y)
        print(saveFolder + "/raw_" + str(inner_index) + ".jpg")
        img.save(saveFolder + "/raw_" + str(inner_index) + ".jpg")

    def save_samples(self, parent_folder, task_name):
        self.safe_create_dir(parent_folder + "/" + task_name + "/" + self.traj_name)
        # self.safe_create_dir(parent_folder + "/" + task_name + "/all")
        total_count = 1
        trajLens = []
        for i in range(len(self.all_trajs)):
            trajLens.append(len(self.all_trajs[i]))
        # find the minimum len
        avgLen = np.min(np.array(trajLens))
        print("adjust to step num: " + str(avgLen))
        for i in range(len(self.all_trajs)):
            this_folder = parent_folder + "/" + task_name + "/" + self.traj_name + "/" + str(i+1)
            self.safe_create_dir(this_folder)
            c = self.all_trajs[i]
            gap = trajLens[i]/avgLen
            self.save_img(c, 0, this_folder, 1)
            for j in range(avgLen-2):
                this_index = int((j+1) * gap)
                self.save_img(c, this_index, this_folder, j+2)
                total_count +=1
            self.save_img(c, len(c)-1, this_folder, avgLen)

    def gen_test_seqs(self, trajs, save_folder):
        return

    def safe_create_dir(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
