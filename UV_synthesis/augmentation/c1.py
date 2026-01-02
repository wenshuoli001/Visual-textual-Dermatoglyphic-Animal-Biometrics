'''
e1
'''

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def localTranslationWarpFastWithStrength(srcImg, startX, startY, endX, endY, radius, strength):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    maskImg = np.zeros(srcImg.shape[:2], np.uint8)
    cv2.circle(maskImg, (startX, startY), math.ceil(radius), (255, 255, 255), -1)

    K0 = 100 / strength

    # 计算公式中的|m-c|^2
    ddmc_x = (endX - startX) * (endX - startX)
    ddmc_y = (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape

    mapX = np.vstack([np.arange(W).astype(np.float32).reshape(1, -1)] * H)
    mapY = np.hstack([np.arange(H).astype(np.float32).reshape(-1, 1)] * W)

    distance_x = (mapX - startX) * (mapX - startX)
    distance_y = (mapY - startY) * (mapY - startY)
    distance = distance_x + distance_y
    K1 = np.sqrt(distance)
    ratio_x = (ddradius - distance_x) / (ddradius - distance_x + K0 * ddmc_x)
    ratio_y = (ddradius - distance_y) / (ddradius - distance_y + K0 * ddmc_y)
    ratio_x = ratio_x * ratio_x
    ratio_y = ratio_y * ratio_y

    UX = mapX - ratio_x * (endX - startX) * (1 - K1 / radius)
    UY = mapY - ratio_y * (endY - startY) * (1 - K1 / radius)

    np.copyto(UX, mapX, where=maskImg == 0)
    np.copyto(UY, mapY, where=maskImg == 0)
    UX = UX.astype(np.float32)
    UY = UY.astype(np.float32)
    copyImg = cv2.remap(srcImg, UX, UY, interpolation=cv2.INTER_LINEAR)

    return copyImg


#连续变换并保存j次图像
for j in range(5):

    # 读取图像，包括透明度通道
    img = cv2.imread('original convergence/c1.png', cv2.IMREAD_UNCHANGED)

    # 定义原始关键点
    src_points = np.int_([[102,47],[80,273],[134,307],[72,214]])

    #图像中心坐标
    center = (99, 188)

    # 生成随机值
    '''此处随机值的大小代表了液化的移动程度'''
    rand_val = np.random.randint(1, 30)

    # 计算新的关键点，并转换为int
    dst_points = (src_points + np.random.randint(-rand_val, rand_val, src_points.shape)).astype(np.int_)

    #对每个关键点使用localTranslationWarpFastWithStrength函数
    for i in range(len(src_points)):
        img = localTranslationWarpFastWithStrength(img, src_points[i][0], src_points[i][1], dst_points[i][0], dst_points[i][1], 50, 200)


    #将变换后的新的关键点转换为np.float32
    src_points = dst_points.astype(np.float32)

    #定义矩阵变换关键点的新的随机值
    '''此处随机值的大小代表了仿射变换的程度'''
    rand_val = np.random.randint(1, 3)

    #计算新的关键点，并转换为float32
    dst_points = (src_points + np.random.randint(-rand_val, rand_val, src_points.shape)).astype(np.float32)


    #计算关键点的最大垂直距离,也就是y坐标的最大值减去最小值
    max_y = np.max(dst_points[:,1])
    min_y = np.min(dst_points[:,1])
    y_distance = max_y - min_y
    print('dst_points=',dst_points)
    print('y_distance=',y_distance)


    # 计算仿射变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 对图像进行扭曲，边界外的值设为透明
    warped_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), borderValue=(0,0,0,0))

    # 生成随机旋转角度和缩放因子
    '''旋转角度和缩放因子'''
    angle = np.random.uniform(0, 0)  # 旋转角度
    #scale = np.random.uniform(0.6, 1.2)  # 缩放因子
    scale = 0.3


    #计算新的y_distance,计算方法为原来的y)distance乘上cos(angle)再乘上scale
    y_distance = y_distance * math.cos(math.radians(angle)) * scale
    #将y_distance转换为int
    y_distance = int(y_distance)

    print('angle=',angle)
    print('scale=',scale)
    print('y_distance=',y_distance)

    # 计算旋转缩放矩阵
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 对扭曲后的图像进行旋转和缩放`
    final_img = cv2.warpAffine(warped_img, rot_matrix, (warped_img.shape[1], warped_img.shape[0]), borderValue=(0,0,0,0))

    # 保存扭曲后的图像
    copyImg = final_img

    # 如果图像是RGB格式，将其转换为RGBA
    if copyImg.shape[2] == 3:
        copyImg = cv2.cvtColor(copyImg, cv2.COLOR_RGB2RGBA)

    # 获取所有白色（也可以是其他颜色）的像素点
    white_pixels = (copyImg[..., :3] == [255, 255, 255]).all(axis=2)

    # 将所有白色（也可以是其他颜色）的像素点设置为透明
    copyImg[white_pixels, 3] = 0

    # # 保存图像
    # cv2.imwrite('copyImg.png', copyImg)
    #保存第j次变换后的图像
    cv2.imwrite('stripes_part1/c1' + str(j) + '.png', copyImg)

    #将y_distance保存到文件中
    with open('stripes_part1/c1' + str(j) + '.txt','w') as f:
        f.write(str(y_distance) + '\n')
