# 版本号：1.0
# 修改记录：2023/06/08 完成程序的主要功能：图像的分块FFT、加嵌水印、分块逆FFT
# 作者：李铭辉
# 联系方式：2011266@mail.nankai.edu.com

import sys
import random

cmd = None
seed = 2011266
alpha = 100
block_size = 8
path1 = None
path2 = None
path3 = None

if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 2:
        print('用法: python bwm.py <cmd> [arg...] [opts...]')
        print('  命令:')
        print('    encode <image> <watermark> <image(encoded)>')
        print('           图像 + 水印 -> 编码后的图像')
        print('    decode <image> <image(encoded)> <watermark>')
        print('           图像 + 编码后的图像 -> 水印')
        print('  选项:')
        print('    --seed <int>，手动设置随机种子（默认为2011266）')
        print('    --blocksize <int>，手动设置块大小（默认为8）')
        print('    --alpha <float>，手动设置alpha值（默认为100）')
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd != 'encode' and cmd != 'decode':
        print('错误的命令 %s' % cmd)
        sys.exit(1)
    if '--seed' in sys.argv:
        p = sys.argv.index('--seed')
        if len(sys.argv) <= p+1:
            print('缺少 --seed 后面的 <int>')
            sys.exit(1)
        seed = int(sys.argv[p+1])
        del sys.argv[p+1]
        del sys.argv[p]
    if '--blocksize' in sys.argv:
        p = sys.argv.index('--blocksize')
        if len(sys.argv) <= p+1:
            print('缺少 --blocksize 后面的 <int>')
            sys.exit(1)
        block_size = int(sys.argv[p+1])
        del sys.argv[p+1]
        del sys.argv[p]
    if '--alpha' in sys.argv:
        p = sys.argv.index('--alpha')
        if len(sys.argv) <= p+1:
            print('缺少 --alpha 后面的 <float>')
            sys.exit(1)
        alpha = float(sys.argv[p+1])
        del sys.argv[p+1]
        del sys.argv[p]
    if cmd == 'encode':
        if len(sys.argv) < 4:
            print('缺少参数...')
            sys.exit(1)
        path1 = sys.argv[2]
        path2 = sys.argv[3]
        path3 = sys.argv[4]
    else:
        if len(sys.argv) < 3:
            print('缺少参数...')
            sys.exit(1)
        path1 = sys.argv[2]
        path2 = sys.argv[3]

import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft_block(image):
    '''
    对图像进行分块FFT
    '''
    blocks_h = image.shape[0] // block_size
    blocks_w = image.shape[1] // block_size
    fft_blocks = np.zeros(shape=((blocks_h, blocks_w, block_size, block_size)))
    fft_blocks = fft_blocks.astype("complex")
    h_data = np.vsplit(image, blocks_h)
    for h in range(blocks_h):
        block_data = np.hsplit(h_data[h], blocks_w)
        for w in range(blocks_w):
            fft_blocks[h, w, ...] = np.fft.fft2(block_data[w])
    return fft_blocks

def embed(blocks, watermark):
    '''
    将分块FFT的结果嵌入水印
    '''
    embed_blocks = blocks.copy()
    for h in range(watermark.shape[0]):
        for w in range(watermark.shape[1]):
            k = k1 if watermark[h, w] == 1 else k2
            for i in range(block_size):
                embed_blocks[h, w, i, block_size-1-i] = blocks[h, w, i, block_size-1-i] + alpha * k[i]
            for i in range(i, block_size-1):
                embed_blocks[h, w, block_size-i, 1+i] = blocks[h, w, block_size-i, 1+i] + alpha * k[i]
    return embed_blocks

def ifft_block(blocks):
    '''
    对分块进行逆FFT得到图像
    '''
    row = None
    result = None
    h, w = blocks.shape[0], blocks.shape[1]
    for i in range(h):
        for j in range(w):
            block = np.fft.ifft2(blocks[i, j, ...]).real
            row = block if j == 0 else np.hstack((row, block))
        result = row if i == 0 else np.vstack((result, row))
    return result.astype(np.uint8)

def corr2(a, b):
    '''
    计算两个数组的相关程度
    '''
    a = a - np.sum(a) / np.size(a)
    b = b - np.sum(b) / np.size(b)
    np.linalg.norm(b, 2)
    numerator = np.dot(a, b)
    denominator = (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))
    if denominator != 0:
        r = numerator / denominator
    elif numerator > 0:
        r = float("-inf")
    elif numerator < 0:
        r = float("inf")
    elif numerator == 0:
        r = 0.0
    return r

def get_watermark(embed_U_image, watermark_size):
    '''
    提取水印
    '''
    w_h, w_w = watermark_size
    extract_watermark = np.zeros(shape=watermark_size)
    extract_watermark.astype(np.uint8)
    fft_blocks = fft_block(embed_U_image)
    temp = np.zeros(block_size)
    temp = temp.astype("complex")
    for h in range(w_h):
        for w in range(w_w):
            for i in range(block_size):
                temp[i] = fft_blocks[h, w, i, block_size-1-i]
            if corr2(temp, k1) > corr2(temp, k2):
                extract_watermark[h, w] = 255
            else:
                extract_watermark[h, w] = 0
    return extract_watermark.astype(np.uint8)

if cmd == 'encode':
    image_path = path1
    watermark_path = path2
    embed_image_path = path3
    # 第一步、读取水印和图片


    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    watermark = np.where(watermark < np.mean(watermark), 0, 1)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 第二步、检查小块的边长是否合理
    i_h, i_w = image.shape[:2]
    if not(i_h % block_size == 0 and i_w % block_size == 0):
        print("图像的宽度或高度无法被小方块的边长整除！小方块的边长为 {:}，原图的尺寸为 {:}！请重新选择小方块的边长！".format(
            block_size, image.shape))
        sys.exit(1)
    # 第三步、调整水印的宽高
    watermark = cv2.resize(watermark, (i_w // block_size, i_h // block_size), interpolation=cv2.INTER_LINEAR_EXACT)  # 重新设置水印的大小
    # 第四步、提取U层
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    Y_image, U_image, V_image = yuv_image[..., 0], yuv_image[..., 1], yuv_image[..., 2]
    # 第五步、生成两个随机数组，用于编码
    np.random.seed(seed)
    k1 = np.random.randn(block_size)
    k2 = np.random.randn(block_size)
    # 第六步、对U层进行分块FFT
    fft_blocks = fft_block(U_image)
    # 第七步、嵌入水印
    embed_blocks = embed(fft_blocks, watermark)
    # 第八步、对分块进行逆FFT
    embed_U_image = ifft_block(embed_blocks)
    # 第九步、将U层放回，得到带水印的图像
    yuv_image[..., 1] = embed_U_image
    embed_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    plt.imsave(embed_image_path, embed_image)
else:
    embed_image_path = path1
    extract_watermark_path = path2
    # 第一步、读取图片
    embed_image = cv2.imread(embed_image_path)
    embed_image = cv2.cvtColor(embed_image, cv2.COLOR_BGR2RGB)
    # 第二步、提取U层
    embed_U_image = cv2.cvtColor(embed_image, cv2.COLOR_RGB2YUV)[..., 1]
    # 第三步、检查小方块的边长是否合理
    i_h, i_w = embed_U_image.shape  # 获取原图的大小
    if not(i_h % block_size == 0 and i_w % block_size == 0):
        print("图像的宽度或高度无法被小方块的边长整除！小方块的边长为 {:}，原图的尺寸为 {:}！请重新选择小方块的边长！".format(
            block_size, image.shape))
        sys.exit(1)
    # 第四步、生成两个随机数组，用于解码
    np.random.seed(seed)
    k1 = np.random.randn(block_size)
    k2 = np.random.randn(block_size)
    # 第五步、提取水印
    extract_watermark = get_watermark(embed_U_image, (i_h // block_size, i_w // block_size))
    # 第六步、保存提取的水印
    plt.imsave(extract_watermark_path, extract_watermark, cmap='gray')