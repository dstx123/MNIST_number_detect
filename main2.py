# coding=utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt

basic_path = r'E:\code\Project\MNIST_number_detect\dataset'
# 训练集文件
train_images_idx3_ubyte_file = basic_path + r'\train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = basic_path + r'\train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = basic_path + r'\t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = basic_path + r'\t10k-labels.idx1-ubyte'
# 图片是以28*28=784维向量存储，标签是以10维向量存储

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    param idx3_ubyte_file: idx3文件路径
    return: 数据集，n*row*col维np.array对象，n为图片数量
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'  # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(
        image_size) + 'B'  # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # print(images[i])
        offset += struct.calcsize(fmt_image)
        # plt.imshow(images[i],'gray')
        # plt.pause(0.00001)
        # plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    param idx1_ubyte_file: idx1文件路径
    return: 数据集，n*1维np.array对象，n为图片数量，labels的值是0-9
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
        # print(labels[i])
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def pretreat(train_images, test_images):
    train_images_column = train_images.reshape(60000, 784, 1)
    test_images_column = test_images.reshape(10000, 784, 1)

    train_images_2D = train_images_column.reshape(60000, 784)
    test_images_2D = test_images_column.reshape(10000, 784)
    '''转置矩阵'''
    train_images_2DT = train_images_2D.T
    test_images_2DT = test_images_2D.T

    return train_images_2DT, test_images_2DT


def LS_0(train_labels, A, A_test):
    print('开始计算0的参数矩阵\n')
    # 区分0与非0
    for i in range(len(train_labels)):
        if train_labels[i] == 0:
            train_labels[i] = 1
        elif train_labels[i] != 0:
            train_labels[i] = -1  # 5923个0 /60000 约1/10 正确
    
    b = train_labels
    print('进行0的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成0的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到0的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_1(train_labels, A, A_test):
    print('开始计算1的参数矩阵\n')
    # 区分1与非1
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            train_labels[i] = 1
        elif train_labels[i] != 1:
            train_labels[i] = -1
    
    b = train_labels
    print('进行1的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成1的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到1的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_2(train_labels, A, A_test):
    print('开始计算2的参数矩阵\n')
    # 区分2与非2
    for i in range(len(train_labels)):
        if train_labels[i] == 2:
            train_labels[i] = 1
        elif train_labels[i] != 2:
            train_labels[i] = -1
    
    b = train_labels
    print('进行2的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成2的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到2的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_3(train_labels, A, A_test):
    print('开始计算3的参数矩阵\n')
    # 区分3与非3
    for i in range(len(train_labels)):
        if train_labels[i] == 3:
            train_labels[i] = 1
        elif train_labels[i] != 3:
            train_labels[i] = -1
    
    b = train_labels
    print('进行3的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成3的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到3的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_4(train_labels, A, A_test):
    print('开始计算4的参数矩阵\n')
    # 区分4与非4
    for i in range(len(train_labels)):
        if train_labels[i] == 4:
            train_labels[i] = 1
        elif train_labels[i] != 4:
            train_labels[i] = -1
    
    b = train_labels
    print('进行4的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成4的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到4的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_5(train_labels, A, A_test):
    print('开始计算5的参数矩阵\n')
    # 区分5与非5
    for i in range(len(train_labels)):
        if train_labels[i] == 5:
            train_labels[i] = 1
        elif train_labels[i] != 5:
            train_labels[i] = -1
    
    b = train_labels
    print('进行5的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成5的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到5的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_6(train_labels, A, A_test):
    print('开始计算6的参数矩阵\n')
    # 区分6与非6
    for i in range(len(train_labels)):
        if train_labels[i] == 6:
            train_labels[i] = 1
        elif train_labels[i] != 6:
            train_labels[i] = -1
    
    b = train_labels
    print('进行6的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成6的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到6的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_7(train_labels, A, A_test):
    print('开始计算7的参数矩阵\n')
    # 区分7与非7
    for i in range(len(train_labels)):
        if train_labels[i] == 7:
            train_labels[i] = 1
        elif train_labels[i] != 7:
            train_labels[i] = -1
    
    b = train_labels
    print('进行7的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成7的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到7的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_8(train_labels, A, A_test):
    print('开始计算8的参数矩阵\n')
    # 区分8与非8
    for i in range(len(train_labels)):
        if train_labels[i] == 8:
            train_labels[i] = 1
        elif train_labels[i] != 8:
            train_labels[i] = -1
    
    b = train_labels
    print('进行8的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成8的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到8的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def LS_9(train_labels, A, A_test):
    print('开始计算9的参数矩阵\n')
    # 区分9与非9
    for i in range(len(train_labels)):
        if train_labels[i] == 9:
            train_labels[i] = 1
        elif train_labels[i] != 9:
            train_labels[i] = -1
    
    b = train_labels
    print('进行9的QR分解中...')
    q, r = np.linalg.qr(A)
    print('已完成9的QR分解')
    print('\n')
    x = np.linalg.pinv(r).dot(q.T.dot(b))
    print('得到9的参数矩阵为：\n',x)
    # print(np.shape(x))  # 494*1
    result_test = A_test.dot(x)
    result_test = result_test.reshape(10000, 1)
    return result_test  # 10000*1，>0为该数字 or <0不为该数字


def test_pic_show(test_images, result_test):
    plt.figure()
    for i in range(10000):
        print(result_test[i])
        maxnum_index = result_test[i].argmax()  # 求得最大值索引
        maxnum = result_test[i].max()  # 求得最大值
        print(maxnum)
        print("该数字为", maxnum_index)
        
        plt.imshow(test_images[i])
        # plt.pause(0.00001)
        plt.show()


def result_analyse(labels, result):
    right_num = 0
    for i in range(len(labels)):
        maxnum_index = result[i].argmax()  # 求得最大值索引
        if maxnum_index == labels[i]:
            right_num = right_num + 1
    error = right_num / 10000 * 100
    print("\n================================")
    print('测试集正确率为', "%.3f" % error, '%')
    print("================================\n")


if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    [train_images_2DT, test_images_2DT] = pretreat(train_images, test_images)

    '''
    数字区域提取：数字一般都在图片中心区域。
    对于训练集60000张图片来说，取它的1%即为600张，
    如果把60000张图片一层一层堆叠在一起，用一根针垂直扎进去，
    如果有超过600张图片该区域非0（即图像里不是黑色的，比如0.5,0.6等等），
    也就是超过59400张该区域是有实际图像的像素值的时，那么就提取这样的一个特征。
    一共有28 × 28 = 784个位置，经过这种方法提取后，有493个特征位置。
    提取非0区域的个数可以采用零范数np.linalg.norm
    '''
    tt = 0
    index = []
    train_image_feature = np.zeros([493, 60000])
    test_image_feature = np.zeros([493, 10000])

    for i in range(784):
        non_zero = np.linalg.norm(train_images_2DT[i, :], ord=0)
        if non_zero >= 600:
            train_image_feature[tt, :] = train_images_2DT[i, :]
            tt = tt + 1
            index.append(i)
    # print(tt)  # 计算出为493
    test_image_feature = test_images_2DT[index, :]

    A = np.hstack([np.ones([60000, 1]), train_image_feature.T])  # 图片像素值矩阵
    A_test = np.hstack([np.ones([10000, 1]), test_image_feature.T])
    
    result_test_0 = LS_0(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_1 = LS_1(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_2 = LS_2(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_3 = LS_3(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_4 = LS_4(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_5 = LS_5(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_6 = LS_6(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_7 = LS_7(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_8 = LS_8(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test_9 = LS_9(train_labels.copy(), A, A_test)  # 10000*1，>0为该数字 or <0不为该数字
    result_test = np.hstack([result_test_0,result_test_1,result_test_2,result_test_3,result_test_4,
                            result_test_5,result_test_6,result_test_7,result_test_8,result_test_9])
    # print(result_test)
    result_analyse(test_labels.copy(), result_test.copy())
    test_pic_show(test_images, result_test.copy())