import math
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def rotated_rectangle(img, rotated_rect, color, lineType=cv2.LINE_8, shift=0):
    (x,y), (width, height), angle = rotated_rect
    angle = math.radians(angle)

    # 回転する前の矩形の頂点
    pt1_1 = (int(x + width / 2), int(y + height / 2))
    pt2_1 = (int(x + width / 2), int(y - height / 2))
    pt3_1 = (int(x - width / 2), int(y - height / 2))
    pt4_1 = (int(x - width / 2), int(y + height / 2))

    # 変換行列
    t = np.array([[np.cos(angle),   -np.sin(angle), x-x*np.cos(angle)+y*np.sin(angle)],
                    [np.sin(angle), np.cos(angle),  y-x*np.sin(angle)-y*np.cos(angle)],
                    [0,             0,              1]])

    tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
    tmp_pt1_2 = np.dot(t, tmp_pt1_1)
    pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))

    tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
    tmp_pt2_2 = np.dot(t, tmp_pt2_1)
    pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))

    tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
    tmp_pt3_2 = np.dot(t, tmp_pt3_1)
    pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))

    tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
    tmp_pt4_2 = np.dot(t, tmp_pt4_1)
    pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))

    points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])
    cv2.fillPoly(img, [points], color, lineType, shift)
    return img

def color_rotate_noise_5_5(w, h, r_w, r_h, out_dir, c1, c2):
    '''
    create rotated rectangles on noise whise size is 5*5
    '''
    # output directory
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # define center
    center_format = np.zeros((5,5))
    center_list = []
    center_num = 0
    for line in range(center_format.shape[0]):
        for row in range(center_format.shape[1]):
            center_list.append(np.array([48*(line+1)-16, 48*(row+1)-16]))
            center_num+=1

    # define "pop-out" position
    popout_list = np.zeros(center_format.shape[0]*center_format.shape[1])
    popout = np.random.randint(0,center_format.shape[0]*center_format.shape[1])
    popout_list[popout] = 1

    # definr "control" position
    control_list = np.zeros(center_format.shape[0]*center_format.shape[1])
    for i in range(control_list.shape[0]):
        control_list[i] = np.random.randint(2)

    # create images and save them
    out_dir = os.path.join(out_dir,"/{}_{}/").format(c1,c2)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    image_list = [c1, c2, "control"]
    color_dic = {"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255)}
    for image in image_list:
        if image==c1:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i in range(popout_list.shape[0]):
                x = int(center_list[i][0])
                y = int(center_list[i][1])
                posision = (x, y)
                rotated_rect = (posision, (r_w, r_h), np.random.randint(0,360))
                if int(popout_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
        elif image==c2:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i in range(popout_list.shape[0]):
                x = int(center_list[i][0])
                y = int(center_list[i][1])
                posision = (x, y)
                rotated_rect = (posision, (r_w, r_h), np.random.randint(0,360))
                if int(popout_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
        else:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i in range(popout_list.shape[0]):
                x = int(center_list[i][0])
                y = int(center_list[i][1])
                posision = (x, y)
                rotated_rect = (posision, (r_w, r_h), np.random.randint(0,360))
                if int(control_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))

def color_rotate_noise_6_6(w, h, r_w, r_h, out_dir, c1, c2):
    '''
    create rotated rectangles on noise whise size is 6*6
    '''
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # define center
    center_format = np.zeros((6,6))
    center_list = []
    center_num = 0
    for line in range(center_format.shape[0]):
        for row in range(center_format.shape[1]):
            center_list.append(np.array([40*(line+1)-12, 40*(row+1)-12]))
            center_num += 1

    # define "pop-out" position
    popout_list = np.zeros(center_format.shape[0]*center_format.shape[1])
    popout = np.random.randint(0,center_format.shape[0]*center_format.shape[1])
    popout_list[popout] = 1

    # define "control" position (random N and O)
    control_list = np.zeros(center_format.shape[0]*center_format.shape[1])
    for i in range(control_list.shape[0]):
        control_list[i] = np.random.randint(2)

    # crate images and save them
    out_dir = os.path.join(out_dir,"/{}_{}/").format(c1,c2)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    image_list = [c1, c2, "control"]
    color_dic = {"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255)}
    for image in image_list:
        if image==c1:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i in range(popout_list.shape[0]):
                x = int(center_list[i][0])
                y = int(center_list[i][1])
                posision = (x, y)
                rotated_rect = (posision, (r_w, r_h), np.random.randint(0,360))
                if int(popout_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
        elif image==c2:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i in range(popout_list.shape[0]):
                x = int(center_list[i][0])
                y = int(center_list[i][1])
                posision = (x, y)
                rotated_rect = (posision, (r_w, r_h), np.random.randint(0,360))
                if int(popout_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
        else:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i in range(popout_list.shape[0]):
                x = int(center_list[i][0])
                y = int(center_list[i][1])
                posision = (x, y)
                rotated_rect = (posision, (r_w, r_h), np.random.randint(0,360))
                if int(control_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))

def random_color_rotated_noise(w, h, r_w, r_h, object_num, out_dir, c1, c2):
    '''
    create rotated rectangles on noise which are scattered
    '''
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # define "pop-out" position
    popout_list = np.zeros(object_num)
    popout = np.random.randint(0,object_num)
    popout_list[popout] = 1

    # define "control" position (random N and O)
    control_list = np.zeros(object_num)
    for i in range(object_num):
        control_list[i] = np.random.randint(2)

    # color definition
    color_dic = {"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255)}

    # caliculate rectangle's diagonal with its shape
    diagonal = np.sqrt(r_w**2 + r_h**2)

    # define rectangle's arrengement
    obj_num = 0
    obj_list = []
    while obj_num <= 24:
        # difine center position
        if obj_num == 0:
            base_image = np.zeros((h,w,3), dtype=np.uint8)
            x = np.random.randint(diagonal, w-diagonal)
            y = np.random.randint(diagonal, h-diagonal)
            posision = (x, y)
            angle = np.random.randint(0,360)
            rotated_rect = (posision, (r_w, r_h), angle)
            if popout_list[obj_num] == 0:
                rotated_rectangle(base_image, rotated_rect, color_dic[c1])
            else:
                rotated_rectangle(base_image, rotated_rect, color_dic[c2])
            obj_list.append([posision,angle])
            obj_num+=1
            base_image = np.sum(base_image, axis=2)
        else:
            current_image = np.zeros((h,w,3), dtype=np.uint8)
            x = np.random.randint(diagonal, w-diagonal)
            y = np.random.randint(diagonal, h-diagonal)
            posision = (x, y)
            angle = np.random.randint(0,360)
            rotated_rect = (posision, (r_w, r_h), angle)
            if popout_list[obj_num] == 0:
                rotated_rectangle(current_image, rotated_rect, color_dic[c1])
            else:
                rotated_rectangle(current_image, rotated_rect, color_dic[c2])
            current_image = np.sum(current_image, axis=2)
            test = base_image + current_image
            if int(np.max(test)) <= 255:
                obj_list.append([posision,angle])
                base_image = base_image + current_image
                obj_num+=1
            else:
                pass

    # create images and save them
    out_dir = os.path.join(out_dir,"/{}_{}/").format(c1,c2)
    image_list = [c1, c2, "control"]
    color_dic = {"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255)}
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for image in image_list:
        if image==c1:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i,obj in enumerate(obj_list):
                rotated_rect = (obj[0],(r_w,r_h),obj[1])
                if int(popout_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
        elif image==c2:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i,obj in enumerate(obj_list):
                rotated_rect = (obj[0],(r_w,r_h),obj[1])
                if int(popout_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
        else:
            base_image = np.random.randint(0,256,size=(h,w,3), dtype=np.uint8)
            for i,obj in enumerate(obj_list):
                rotated_rect = (obj[0],(r_w,r_h),obj[1])
                if int(control_list[i]) == 0:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c1])
                else:
                    rotated_rectangle(base_image, rotated_rect, color_dic[c2])
            cv2.imwrite(out_dir+"/{}.png".format(image), base_image)
            plt.figure()
            plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
