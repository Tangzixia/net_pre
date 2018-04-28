import numpy as np
import cv2
import queue
import os
import threading
import _thread
import time
import random

class data_reader():
    def __init__(self, img_path, class_path,pre_size,pre_size_w,pre_size_h, batch_size):
        self.img_path = img_path
        self.class_path = class_path
        self.batch_size = batch_size
        self.pre_size_w=pre_size_w
        self.pre_size_h=pre_size_h
        self.data_queue256_256 = queue.Queue(3 * batch_size)
        key_img_names = os.listdir(img_path)
        self.key_names = []
        for tmp_name in key_img_names:
            self.key_names.append(tmp_name.split(".")[0])
        self.thread_lock = threading.Lock
        self.labels=self.read_all_classes(class_path)

        for t in range(4):
            _thread.start_new_thread(self.single_thread_fuc,())

    # def read_single_data(self, key_name):
    #     # tmp_img = cv2.imread(os.path.join(self.img_path, key_name+".png"))
    #     # tmp_class = np.load(os.path.join(self.class_path, key_name+".npy"))
    #
    #     return tmp_img, tmp_class
    def read_all_classes(self,label_file):
        dict_={}
        with open(label_file,'r') as f:
            lines=f.readlines()
            lines_=[(line.split(" ")[0].split(".")[0],line.split(" ")[1]) for line in lines]
            for line in lines_:
                dict_[line[0]]=int(line[1])
        return dict_

    def read_single_data(self,key_name):
        tmp_img=cv2.resize(cv2.imread(os.path.join(self.img_path,key_name+".png")),(self.pre_size_w,self.pre_size_h))
        tmp_label=self.labels[key_name]
        return tmp_img,tmp_label

    def single_thread_fuc(self):
        while(True):
            tmp_key_index = np.random.random_integers(0, len(self.key_names) - 1)
            tmp_data = self.read_single_data(self.key_names[tmp_key_index])
            if self.data_queue256_256.qsize() < 2*self.batch_size:
                self.data_queue256_256.put(tmp_data)
    def one_hot_code(self,data,num_class):
        data_ret=np.zeros((len(data),num_class),dtype=np.float32)
        for id,item in enumerate(data):
            data_ret[id][item]=1.0
        return data_ret
    def read_data(self):
        tmp_imgs = []
        tmp_class_s = []
        while True:
            if self.data_queue256_256.qsize()<self.batch_size:
                continue
            else:
                break
        for i in range(self.batch_size):
            tmp_data = self.data_queue256_256.get()
            tmp_imgs.append(tmp_data[0])
            tmp_class_s.append(tmp_data[1])
        random.shuffle(tmp_imgs)
        random.shuffle(tmp_class_s)
        tmp_imgs = np.asarray(tmp_imgs)
        tmp_class_s = self.one_hot_code(tmp_class_s,2)
        return tmp_imgs, tmp_class_s

#class_path = "C:\\Users\\Administrator\\Desktop\\lable.txt"
def check(class_path):
    with open(class_path,'r') as f:
        lines=f.readlines()
        list=[]
        lines=[line.split(".")[0] for line in lines]
        count=1
        for line in lines:
            print(count)
            count = count + 1
            if line not in list:
                list.append(line)
            else:
                continue
        print(len(list))
