import random
from PIL import Image
import json

# 创建列表和字典
'''条纹种类'''
re = [0, 1, 2, 3]
'''条纹种类对应的字典'''
re_dict = {0: 'r', 1: 'e', 2: 'b', 3: 'c'}
'''每种条纹的类型个数'''
first_index = {'r': [ 3, 4, 7, 8], 'e': [1, 2, 3, 4, 5, 6, 7, 8], 'b': [1, 2, 3,5,6,7], 'c': [1, 2, 3, 4]}
second_index = [0, 1, 2, 3, 4]#每个图像变换出5个新的图像
image_list = []
position_list = []
name_list = []
'''每种条纹的位置'''
position_dict = {'r1': [60, 83], 'r2': [67, 96], 'r3': [144, 298], 'r4': [113, 130], 'r5': [59, 103], 'r6': [47, 84], 'r7': [74, 78], 'r8': [84, 126], 'e1': [95, 127], 'e2': [134, 163], 'e3': [127, 179], 'e4': [173, 297], 'e5': [70, 150], 'e7': [26, 126], 'e8': [16, 62], 'e6': [103, 130], 'b1': [129, 222], 'b2': [126, 244], 'b3': [63, 81], 'b4': [49, 72],'b5': [23, 94],'b6': [24, 58],'b7': [21, 62], 'c1': [99, 188], 'c2': [136,228], 'c3': [30,94], 'c4': [30,94]}

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

'''定义生成的图像数量'''
image_number = 2000
for j in range(image_number):

    '''定义密度'''
    x1,y1 = 1559,400
    z1 = 3200
    interval = random.randint(350, 450)
    #interval = 250
    interval_vertical0 = 150
    interval_vertical1 = 500
    interval_horizontal = 400
    scale = int(interval/80)
    coordinates_list0 = generate_coordinates(x1, y1, z1, interval)
    coordinates_list1 =generate_coordinates(int(x1+0.5*interval), y1+interval_vertical0, int(z1-0.5*interval), interval)

    coordinates_list2 = generate_coordinates(x1+interval_horizontal, y1+interval_vertical1, z1, interval)
    coordinates_list3 =generate_coordinates(int(x1+interval_horizontal+0.5*interval), y1+interval_vertical1+interval_vertical0, int(z1-0.5*interval), interval)
    coordinates_list = coordinates_list0 + coordinates_list1+coordinates_list2+coordinates_list3





    '''背景中点的位置'''
    #background_positions = [[1559,366],[1703,726],[1872,377],[2018,728],[2181,377],[2348,732],[2496,392],[2654,732],[2817,400],[2974,737],[3113,413]]
    background_positions = coordinates_list
    #计算background_positions的长度

    stripes_number = len(background_positions)
    # print('stripes_number= ', stripes_number)
    # print('len(coordinates_list0)=', len(coordinates_list0))
    # print(coordinates_list0)
    # print('len(coordinates_list1)=', len(coordinates_list1))
    # print(coordinates_list1)
    # print('len(coordinates_list2)=', len(coordinates_list2))
    # print(coordinates_list2)
    # print('len(coordinates_list3)=', len(coordinates_list3))
    # print(coordinates_list3)


    # 选择六个条纹图像
    for _ in range(stripes_number):
        '''每种类型条纹的出现频率'''
        re_choice = random.choices(re, weights=[0.65, 0.11, 0.21, 0.03], k=1)[0]
        #re_choice = random.choices(re, weights=[0.3, 0.3, 0.3, 0.1], k=1)[0]
        name = re_dict[re_choice]
        name_list.append(name)
        first_choice = random.choice(first_index[name])
        name += str(first_choice)
        position = position_dict[name]
        second_choice = random.choice(second_index)
        name += str(second_choice) + '.png'
        image_list.append(name)
        position_list.append(position)

    #print(image_list)
    #print(position_list)

    # 导入背景图片
    background = Image.open('patch_one_background.png').convert('RGBA')

    # 对六个点加入随机水平移动和随机竖直移动
    new_positions = [[x[0] + random.randint(-0, 0), x[1] - 20 + random.randint(-20, 20)] for x in background_positions]

    import numpy as np

    # 初始化一个列表来保存缩放值
    resize_scales = [0] * stripes_number

    #初始化一个列表来保存长度
    lengths = [0] * stripes_number

    # 导入条纹图片并粘贴到背景图片上
    for i in range(stripes_number):
        stripe = Image.open('stripes_part1/' + image_list[i])
        stripe = stripe.convert('RGBA')
        # 读取与条纹图片同名的txt文件,并将其记录到lengths中
        with open('stripes_part1/' + image_list[i][:-4] + '.txt', 'r') as f:
            lengths[i] = int(f.read())

        #根据i的值来决定缩放值
        if i < 0.5*len(background_positions):
            resize_scale = np.random.uniform(scale, scale+1)
            resize_scales[i] = resize_scale
        else:
            resize_scale = 2*scale+1 - resize_scales[i - int(0.5*len(background_positions))]
            # 记录缩放值
            resize_scales[i] = resize_scale


        # print('resize_scale=',resize_scale)
        # print('resize_scales=',resize_scales)

        #设置每个图像的旋转角度,整体旋转
        angle = np.random.uniform(5, 7)

        #对图像进行旋转
        stripe = stripe.rotate(angle)

        #对个别图像进行旋转
        # if i == 11 or i == 12:
        #     stripe = stripe.rotate(5)


        # 对图片进行缩放
        new_size = (int(stripe.size[0] * resize_scale), int(stripe.size[1] * resize_scale))
        stripe = stripe.resize(new_size)

        #使用图像大小的一半作为中心点
        new_center = (int(new_size[0] / 2), int(new_size[1] / 2))


        # 计算新的中心点
        #new_center = (int(round(position_list[i][0] * resize_scale)), int(round(position_list[i][1] * resize_scale)))


        # 创建一个新的空白图片，大小和背景图片一样
        temp_img = Image.new('RGBA', background.size)

        # 将条纹图片粘贴到空白图片的指定位置
        temp_img.paste(stripe, (new_positions[i][0] - new_center[0], new_positions[i][1] - new_center[1]))

        # 使用alpha_composite将条纹图片和背景图片叠加
        background = Image.alpha_composite(background, temp_img)




    #计算新的长度
    new_lengths = [int(lengths[i] * resize_scales[i]) for i in range(stripes_number)]


    # 保存结果
    background.save('results/result' +str(j) + '.png')

    #将image_list中的所有元素去掉后面的.png
    image_list = [x[:-4] for x in image_list]

    # 创建一个字典，将image_list的元素和new_positions和new_lengths的元素一一对应
    data = list(zip(image_list, new_positions, new_lengths))

    data = [{'Image': image, 'Position': position, 'Length': length} for image, position, length in data]

    # 将字典保存为json文件
    with open('results/result' +str(j) + '.json', 'w') as f:
        json.dump(data, f)


    # make the list of name
    #list of the first half



    with open('results/up/' +str(j) + '.txt', 'w') as f:
        name_list1 = name_list[:len(coordinates_list0)+len(coordinates_list1)]
        namelist1_re = []
        #print(name_list1)
        for c in range(len(coordinates_list1)):
            #print('c=', c)
            namelist1_re.append(name_list1[c])
            namelist1_re.append(name_list1[c+len(coordinates_list0)])
        namelist1_re.append(name_list1[c+1])
        #print(namelist1_re)
        #print(name_list)
        f.write(str(namelist1_re))


    with open('results/down/' +str(j) + '.txt', 'w') as f:
        name_list2 = name_list[len(coordinates_list0)+len(coordinates_list1):]
        namelist2_re = []
        #print(name_list2)
        for b in range(len(coordinates_list3)):
            #print('b=', b)
            namelist2_re.append(name_list2[b])
            namelist2_re.append(name_list2[b+len(coordinates_list2)])
        namelist2_re.append(name_list2[b+1])
        #print(namelist2_re)
        #print(name_list)
        f.write(str(namelist2_re))

    image_list = []
    position_list = []
    name_list = []