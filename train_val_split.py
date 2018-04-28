#coding=utf-8
import os
import random
import shutil

img_path="H:/new_CNN_split/imgs"
dest_path="H:/new_CNN_split/imgs_val"
label_path="H:/new_CNN_split/lable.txt"
label_train_path="H:/new_CNN_split/label_train.txt"
label_val_path="H:/new_CNN_split/label_val.txt"
def train_val_split(img_path,dest_path,label_path,label_train_path,label_val_path):
    val_files=random.sample(os.listdir(img_path),5000)
    if os.path.exists(dest_path)==False:
        os.mkdir(dest_path)

    for val_file in val_files:
        shutil.copyfile(os.path.join(img_path,val_file),os.path.join(dest_path,val_file))
        os.remove(os.path.join(img_path,val_file))

    with open(label_path,'r') as f:
        lines=f.readlines()
        dict={}
        line_handle=[(line.split(" ")[0],line.split(" ")[1]) for line in  lines]
        for item in line_handle:
            dict[item[0]]=item[1]

    with open(label_train_path,'w') as f:
        for key,val in dict.items():
            if key not in val_files:
                f.write(key+" "+val)

    with open(label_val_path,'w') as f:
        for item in val_files:
            f.write(item+" "+dict[item])

#临时救场的函数
def help_transfer(label_path,dest_path):
    with open(label_path,'r') as f:
        lines=f.readlines()
        list=[]
        for line in lines:
            if line!="\n":
                list.append(line)
    with open(dest_path,'w') as f:
        for item in list:
            f.write(item)