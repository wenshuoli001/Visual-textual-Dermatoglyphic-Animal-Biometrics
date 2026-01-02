import random
from PIL import Image
import json
import numpy as np
import os

# 创建列表和字典
'''条纹种类'''
re = [0, 1, 2, 3]
re_dict = {0: 'r', 1: 'e', 2: 'b', 3: 'c'}
'''每种条纹的类型个数'''
first_index1 = {'r': [1, 2,4], 'e': [1, 2, 3, 5], 'b': [1,5], 'c': [3, 4, 5]}
second_index = [0, 1, 2, 3, 4]#每个图像变换出5个新的图像
first_index2 = {'r': [3], 'e': [4,6], 'b': [ 2, 3, 4], 'c': [1, 2]}
image_list = []
name_list = []
position_list = []
'''每种条纹的位置'''
position_dict = {'r1': [60, 83], 'r2': [67, 96], 'r3': [144, 298], 'r4': [113, 130], 'r5': [59, 103], 'r6': [47, 84], 'r7': [74, 78], 'r8': [84, 126], 'e1': [95, 127], 'e2': [134, 163], 'e3': [127, 179], 'e4': [173, 297], 'e5': [70, 150], 'e6': [103, 130], 'b1': [129, 222], 'b2': [126, 244], 'b3': [63, 81], 'b4': [49, 72], 'b5': [28,123], 'c1': [129, 222], 'c2': [129, 222]}

'''生成坐标'''
def generate_coordinates(x, y, z, interval):
    coordinates = []
    if x < z:
        current_x = x
        while current_x <= z:
            coordinates.append([current_x, y])
            current_x += interval
    else:
        current_x = x
        while current_x >= z:
            coordinates.append([current_x, y])
            current_x -= interval
    return coordinates





results_list = []
results_directory = 'results'
for filename in os.listdir(results_directory):
    if filename.endswith('.png'):
        background = Image.open(results_directory + '/' + filename).convert('RGBA')

        '''定义密度'''
        x1, y1 = 3200, 700
        z1 = 4000
        #interval = random.randint(130, 180)
        interval = 170

        scale = int(interval / 18)
        coordinates_list0 = generate_coordinates(x1, y1, z1, interval)

        coordinates_list = coordinates_list0

        # '''背景中点的位置'''
        # background_positions = [[3449,565],[3577,565],[3741,573],[3893,573],[3513,909],[3673,913],[3809,909]]
        # #计算background_positions的长度
        background_positions = coordinates_list
        stripes_number = len(background_positions)
        print('stripes_number= ', stripes_number)

        # 对于当前路径下results文件夹中的所有png文件,将其作为背景

        # 选择六个条纹图像
        for n in range(stripes_number):
            '''每种类型条纹的出现频率'''
            if n < 0.5*stripes_number:
                re_choice = random.choices(re, weights=[0.58, 0.13, 0.22, 0.07], k=1)[0]
                name = re_dict[re_choice]
                name_list.append(name)
                first_choice = random.choice(first_index1[name])
                name += str(first_choice)
                position = position_dict[name]
                second_choice = random.choice(second_index)
                name += str(second_choice) + '.png'
                image_list.append(name)
                position_list.append(position)
            else:
                re_choice = random.choices(re, weights=[0.58, 0.13, 0.22, 0.07], k=1)[0]
                name = re_dict[re_choice]
                name_list.append(name)
                first_choice = random.choice(first_index2[name])
                name += str(first_choice)
                position = position_dict[name]
                second_choice = random.choice(second_index)
                name += str(second_choice) + '.png'
                image_list.append(name)
                position_list.append(position)

        # 对六个点加入随机水平移动和随机竖直移动
        new_positions = [[x[0] + random.randint(0, 0), x[1] - 50 + random.randint(-1, 1)] for x in background_positions]


        # 初始化一个列表来保存缩放值
        resize_scales = [0] * stripes_number

        # 初始化一个列表来保存长度
        lengths = [0] * stripes_number

        # 导入条纹图片并粘贴到背景图片上
        for i in range(stripes_number):
            stripe = Image.open('stripes_back/' + image_list[i])
            stripe = stripe.convert('RGBA')
            # 读取与条纹图片同名的txt文件,并将其记录到lengths中
            with open('stripes_back/' + image_list[i][:-4] + '.txt', 'r') as f:
                lengths[i] = int(f.read())

            # 根据i的值来决定缩放值
            if i < 0.5 * len(background_positions):
                resize_scale = np.random.uniform(scale, scale + 1)
                resize_scales[i] = resize_scale
            else:
                resize_scale = 2 * scale + 1 - resize_scales[i - int(0.5 * len(background_positions))]
                # 记录缩放值
                resize_scales[i] = resize_scale

            # # 根据i的值来决定缩放值
            # if i < 4:
            #     resize_scale = np.random.uniform(3.5, 4.5)
            #     resize_scales[i] = resize_scale
            # else:
            #     resize_scale = 8 - resize_scales[i - 4]
            #     # 记录缩放值
            #     resize_scales[i] = resize_scale

            # resize_scale = resize_scale

            # print('resize_scale=',resize_scale)
            # print('resize_scales=',resize_scales)

            #设置每个图像的旋转角度
            angle = np.random.uniform(15, 20)

            #对图像进行旋转
            stripe = stripe.rotate(angle)

            # 对图片进行缩放
            new_size = (int(stripe.size[0] * resize_scale), int(stripe.size[1] * resize_scale))
            stripe = stripe.resize(new_size)

            # 使用图像大小的一半作为中心点
            new_center = (int(new_size[0] / 2), int(new_size[1] / 2))

            # 计算新的中心点
            # new_center = (int(round(position_list[i][0] * resize_scale)), int(round(position_list[i][1] * resize_scale)))

            # 创建一个新的空白图片，大小和背景图片一样
            temp_img = Image.new('RGBA', background.size)

            # 将条纹图片粘贴到空白图片的指定位置
            temp_img.paste(stripe, (new_positions[i][0] - new_center[0], new_positions[i][1] - new_center[1]))

            # 使用alpha_composite将条纹图片和背景图片叠加
            background = Image.alpha_composite(background, temp_img)

        print('resize_scales=', resize_scales)
        print('lengths=', lengths)

        # 计算新的长度
        new_lengths = [int(lengths[i] * resize_scales[i]) for i in range(stripes_number)]

        # 保存结果
        background.save('results1' + '/' + filename[:-4] + '.png')

        # 将image_list中的所有元素去掉后面的.png
        image_list = [x[:-4] for x in image_list]

        # 创建一个字典，将image_list的元素和new_positions和new_lengths的元素一一对应
        data = dict(zip(image_list, zip(new_positions, new_lengths)))

        # 将字典保存为json文件
        with open('results1' + '/' + filename[:-4] + '.json', 'w') as f:
            json.dump(data, f)


        print('name_list',name_list)

        #save the name_list
        with open('results1/back/' + filename[6:-4] + '.txt', 'w') as f:
            f.write(str(name_list))

        image_list = []
        position_list = []
        name_list = []