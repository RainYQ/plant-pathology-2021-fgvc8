import os
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import math
import pandas as pd
import cv2
import csv


TRAIN_DATA_ROOT = "./train_images/"
# 查找并检查重复图片
# pic_name = pd.read_csv("duplicates.csv", encoding='utf-8')

# for i in range(pic_name.shape[0]):
    
#     img = cv2.imread(TRAIN_DATA_ROOT + pic_name["p1"][i])
#     cv2.imwrite("check_images/"+str(i)+"_1.jpg", img)
#     print(pic_name["p1"][i]+"has been saved.")

#     img2 = cv2.imread(TRAIN_DATA_ROOT + pic_name["p2"][i])
#     cv2.imwrite("check_images/"+str(i)+"_2.jpg", img2)
#     print(pic_name["p2"][i]+"has been saved.")

#     print("*********")
# 手动添加是否重复的状态


# 找重复图片的labels
pic_name = pd.read_csv("duplicates_compare_result.csv", encoding='utf-8')
print(pic_name["result"])
count = 0
pic = [[] for i in range(105)]
simipic = [[] for i in range(105)]
for i in range(pic_name.shape[0]):
    if (pic_name["result"][i] == "y"):
        simipic[count].append(pic_name["p1"][i] + " " + pic_name["p2"][i])
        pic[count] = "".join(simipic[count]).split()
        count = count + 1
print(pic)
print(count)

train_pic = pd.read_csv("train.csv", encoding='utf-8')

for i in range(train_pic.shape[0]):
    for j in range(count):
        for k in range(2):
            if (train_pic["image"][i] == pic[j][k]):
                pic[j] = np.append(pic[j], train_pic["labels"][i])

# 不同均删，相同保留其一
for i in range(count):
    if(pic[i][2] == pic[i][3]):
        pic[i] = np.append(pic[i], "keep1")
    else:
        pic[i] = np.append(pic[i], "delete")

#print(pic)

# # 写重复文件
# duplicate_pic = open('duplicate_pics.csv', 'w', encoding = 'utf-8', newline = "")
# csv_writer = csv.writer(duplicate_pic)
# csv_writer.writerow(["p1", "p2", "label1","label2", "op"])
# for i in range(count):
#     csv_writer.writerow(pic[i])
# duplicate_pic.close()
pic = pd.read_csv("duplicate_pics.csv", encoding='utf-8')
# 去掉重复图片
for i in range(count):
    row_indexs = train_pic[train_pic["image"] == pic["p1"][i]].index 
    train_pic.drop(row_indexs, inplace=True)
    if(pic["op"][i] == "delete"):
        row_indexs = train_pic[train_pic["image"] == pic["p2"][i]].index
        train_pic.drop(row_indexs, inplace=True)

print(train_pic)

train_pic.to_csv("train_without_rep.csv", index=False)
