# 卷积神经网络

如果按照Boston房价预测的模型来进行图片的学习，存在以下两个问题：
1. **输入数据的空间信息被丢失。** 像素和其附近的其他像素共同构成的特征（如形状）被一维化之后，或RGB多个通道的数据共同表示的信息被一维化之后，这些模式会被忽略。
1. **模型参数过多，容易发生过拟合。** 在手写数字识别的案例中，每个像素点都要跟所有输出的神经元相连接。当图片尺寸变大时，输入神经元的个数会按图片尺寸的平方增大，导致模型参数过多，容易发生过拟合。

为了解决上述问题，我们引入卷积神经网络进行特征提取，既能提取到相邻像素点之间的特征模式，又能保证参数的个数不随图片尺寸变化。

在卷积神经网络中，计算范围是在像素点的**空间邻域**内进行的，卷积核参数的数目也远小于全连接层。卷积核本身与输入图片大小无关，它代表了对空间邻域内某种特征模式的提取。*比如，有些卷积核提取物体边缘特征，有些卷积核提取物体拐角处的特征，图像上不同区域共享同一个卷积核。* 当输入图片大小不一样时，仍然可以使用同一个卷积核进行操作。


# 卷积 Convolution

## 卷积计算

在卷积神经网络中，卷积层的实现方式实际上是数学中定义的**互相关 （cross-correlation）运算**，与数学分析中的卷积定义有所不同。计算方法是：将卷积核与图片对应区域对应像素相乘后求和，再加上偏置项。卷积核（kernel）也被叫做滤波器（filter），假设卷积核的高和宽分别为 k_h 和 k_w，则称其为  k_h * k_w 卷积。比如 3*5 卷积。

### 填充 padding

为了避免卷积之后图片尺寸变小，通常会在图片的外围进行填充(padding)。

### 步幅 stride

指卷积核在图片上移动时，每次移动的像素点数。其中，宽和高方向的步幅可以不相等，分别为 s_h 和 s_w。

## 感受野 Receptive Field

输出特征图上每个点的数值，是由输入图片上大小为  k_h * k_w 的区域的元素与卷积核每个元素相乘再相加得到的，所以输入图像上  k_h * k_w 区域内**每个元素数值的改变，都会影响输出点的像素值**。我们将这个区域叫做输出特征图上对应点的感受野。感受野内每个元素数值的变动，都会影响输出点的数值变化。

## 多输入通道

当图片拥有多个通道（如RGB三色）时，要计算卷积的输出结果，卷积核的形式也会发生变化。通常将卷积核的输出通道数叫做**卷积核的个数**。设输入通道数为3，则输入数据的形状可以表示为 C_in * H_in * W_in。单输出通道的计算过程为：

1. 对每个通道分别设计一个2维数组作为卷积核，卷积核的形状是 C_in * k_h * k_w。
1. 对每个通道分别用对应卷积核的“层”做卷积计算。
1. 将这 C_in 个通道的计算结果相加，得到形状为 H_out * W_out 的数组。（各通道结果相加）

多输出通道的场景下，免去了最后一步各通道卷积结果的相加步骤，输出特征图维度为 C_out * H_out * W_out。


## 批量操作

在卷积神经网络的计算中，通常将多个样本放在一起形成一个mini-batch进行批量操作，即输入数据为 N * C_in * H_in * W_in。这种情况下，卷积核的维度不变，但输出特征图的维度变为 N * C_out * H_out * W_out。

## 飞桨卷积API介绍

飞桨卷积算子对应的API是paddle.fluid.dygraph.Conv2D，用户可以直接调用API进行计算，也可以在此基础上修改。常用的参数如下：

- num_channels (int) - 输入图像的通道数。
- num_fliters (int) - 卷积核的个数，和输出特征图通道数相同，相当于上文中的C_out。
- filter_size(int|tuple) - 卷积核大小，可以是整数，比如3，表示卷积核的高和宽均为3 ；或者是两个整数的list，例如[3,2]，表示卷积核的高为3，宽为2。
- stride(int|tuple) - 步幅，可以是整数，默认值为1，表示垂直和水平滑动步幅均为1；或者是两个整数的list，例如[3,2]，表示垂直滑动步幅为3，水平滑动步幅为2。
- padding(int|tuple) - 填充大小，可以是整数，比如1，表示竖直和水平边界填充大小均为1；或者是两个整数的list，例如[2,1]，表示竖直边界填充大小为2，水平边界填充大小为1。
- act（str）- 应用于输出上的激活函数，如Tanh、Softmax、Sigmoid，Relu等，默认值为None。

### 卷积算子应用举例

#### 1. 简单的黑白边界检测

设图像左边白色(1)，右边黑色(0)，欲监测黑白分界处，可以设置宽度方向的卷积核为 [1, 0, -1]。当卷积核处在白色或黑色区域时，卷积的结果均为0，只有处在分界线处时，卷积的结果不为0。由此实现了黑白边界的检测，具体代码如下：

```
import matplotlib.pyplot as plt

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.initializer import NumpyArrayInitializer
%matplotlib inline

with fluid.dygraph.guard():
    # 创建初始化权重参数w
    w = np.array([1, 0, -1], dtype='float32')
    # 将权重参数调整成维度为[cout, cin, kh, kw]的四维张量
    w = w.reshape([1, 1, 1, 3])
    # 创建卷积算子，设置输出通道数，卷积核大小，和初始化权重参数
    # filter_size = [1, 3]表示kh = 1, kw=3
    # 创建卷积算子的时候，通过参数属性param_attr，指定参数初始化方式
    # 这里的初始化方式时，从numpy.ndarray初始化卷积参数
    conv = Conv2D(num_channels=1, num_filters=1, filter_size=[1, 3],
            param_attr=fluid.ParamAttr(
              initializer=NumpyArrayInitializer(value=w)))
    
    # 创建输入图片，图片左边的像素点取值为1，右边的像素点取值为0
    img = np.ones([50,50], dtype='float32')
    img[:, 30:] = 0.
    # 将图片形状调整为[N, C, H, W]的形式
    x = img.reshape([1,1,50,50])
    # 将numpy.ndarray转化成paddle中的tensor
    x = fluid.dygraph.to_variable(x)
    # 使用卷积算子作用在输入图片上
    y = conv(x)
    # 将输出tensor转化为numpy.ndarray
    out = y.numpy()

f = plt.subplot(121)
f.set_title('input image', fontsize=15)
plt.imshow(img, cmap='gray')

f = plt.subplot(122)
f.set_title('output featuremap', fontsize=15)
# 卷积算子Conv2D输出数据形状为[N, C, H, W]形式
# 此处N, C=1，输出数据形状为[1, 1, H, W]，是4维数组
# 但是画图函数plt.imshow画灰度图时，只接受2维数组
# 通过numpy.squeeze函数将大小为1的维度消除
plt.imshow(out.squeeze(), cmap='gray')
plt.show()
```
使用以下代码查看卷积层的参数。
```
# 查看卷积层的参数
with fluid.dygraph.guard():
    # 通过 conv.parameters()查看卷积层的参数，返回值是list，包含两个元素
    print(conv.parameters())
    # 查看卷积层的权重参数名字和数值
    print(conv.parameters()[0].name, conv.parameters()[0].numpy())
    # 参看卷积层的偏置参数名字和数值
    print(conv.parameters()[1].name, conv.parameters()[1].numpy())  
```
在真实图片中，可以使用如下卷积核，实现物体外形轮廓的检测。
```
# 设置卷积核参数
    w = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype='float32')/8
    w = w.reshape([1, 1, 3, 3])
    # 由于输入通道数是3，将卷积核的形状从[1,1,3,3]调整为[1,3,3,3]
    w = np.repeat(w, 3, axis=1)
    # 创建卷积算子，输出通道数为1，卷积核大小为3x3，
    # 并使用上面的设置好的数值作为卷积核权重的初始化参数
    conv = Conv2D(num_channels=3, num_filters=1, filter_size=[3, 3], 
            param_attr=fluid.ParamAttr(
              initializer=NumpyArrayInitializer(value=w)))
```

#### 2. 图像均值模糊

另外一种比较常见的卷积核是用当前像素跟它邻域内的像素取平均，这样可以使图像上噪声比较大的点变得更平滑，如下代码所示：

```
import matplotlib.pyplot as plt

from PIL import Image

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.initializer import NumpyArrayInitializer

# 读入图片并转成numpy.ndarray
#img = Image.open('./images/section1/000000001584.jpg')
img = Image.open('./work/images/section1/000000355610.jpg').convert('L')
img = np.array(img)

# 换成灰度图

with fluid.dygraph.guard():
    # 创建初始化参数
    w = np.ones([1, 1, 5, 5], dtype = 'float32')/25
    conv = Conv2D(num_channels=1, num_filters=1, filter_size=[5, 5], 
            param_attr=fluid.ParamAttr(
              initializer=NumpyArrayInitializer(value=w)))
    
    x = img.astype('float32')
    x = x.reshape(1,1,img.shape[0], img.shape[1])
    x = fluid.dygraph.to_variable(x)
    y = conv(x)
    out = y.numpy()

plt.figure(figsize=(20, 12))
f = plt.subplot(121)
f.set_title('input image')
plt.imshow(img, cmap='gray')

f = plt.subplot(122)
f.set_title('output feature map')
out = out.squeeze()
plt.imshow(out, cmap='gray')

plt.show()
```

# 池化 Pooling

池化是**使用某一位置的相邻输出的总体统计特征代替网络在该位置的输出**。当输入数据做出少量平移时，经过池化函数后的大多数输出还能保持不变，可以增加模型的容错性，且有效减小神经元的个数。通常有两种方法，平均池化和最大池化。

- **平均池化**：对池化窗口覆盖区域内的像素取平均值，得到相应的输出特征图的像素值。
- **最大池化**：对池化窗口覆盖区域内的像素取最大值，得到输出特征图的像素值。

当池化窗口在图片上滑动时，会得到整张输出特征图。池化窗口的大小称为池化大小，用 k_h * k_w 表示。在卷积神经网络中用的比较多的是窗口大小为2×2，步幅为2的池化。此外，步幅等定义也与卷积核类似


# ReLu激活函数

在神经网络发展的早期，Sigmoid函数用的比较多，而目前用的较多的激活函数是ReLU。这是因为Sigmoid函数在反向传播过程中，**容易造成梯度的衰减**。

Relu函数在 x<0 时恒为 0，在 x>0 时为 x。

#### 梯度消失现象

在神经网络里面，将经过反向传播之后，梯度值衰减到接近于零的现象称作梯度消失现象。当x为较大的正数。或较小的负数的时候，Sigmoid函数值非常接近于 0，导数也接近于 0。由于最开始是将神经网络的参数随机初始化的，x很有可能取值在数值很大或者很小的区域，如果有多层网络使用了Sigmoid激活函数，则比较靠后的那些层梯度将衰减到非常小的值。

ReLU函数则不同，虽然在 x<0 的地方，ReLU函数的导数为 0。但是在 x≥0 的地方，ReLU函数的导数为 1，能够将y的梯度完整的传递给 x，而不会引起梯度消失。


# 批归一化 Batch Normalization

目的是对神经网络中间层的输出进行**标准化处理**，使得中间层的输出**更加稳定**。

通常我们会对神经网络的数据进行标准化处理，处理后的样本数据集满足**均值为0，方差为1**的统计分布，这是因为当输入数据的分布比较固定时，有利于算法的稳定和收敛。对于深度神经网络来说，由于参数是不断更新的，即使输入数据已经做过标准化处理，但是*对于比较靠后的那些层，其接收到的输入仍然是剧烈变化的，通常会导致数值不稳定，模型很难收敛*。BatchNorm能够使神经网络中间层的输出变得更加稳定，并有如下三个优点：

- 使学习快速进行（能够使用较大的学习率）

- 降低模型对初始值的敏感性

- 从一定程度上抑制过拟合

BatchNorm主要思路是在训练时按mini-batch为单位，对神经元的数值进行归一化，使数据的分布满足均值为0，方差为1。

1. 计算mini-batch内样本的均值。对每个特征分别计算mini-batch内样本的均值。
1. 计算mini-batch内样本的方差。
1. 计算标准化之后的输出。减去平均值，除以标准差（可能需要加上一个小量防止为0）

两个示例：
```
# 输入数据形状是 [N, K]时的示例
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import BatchNorm
# 创建数据
data = np.array([[1,2,3], [4,5,6], [7,8,9]]).astype('float32')
# 使用BatchNorm计算归一化的输出
with fluid.dygraph.guard():
    # 输入数据维度[N, K]，num_channels等于K
    bn = BatchNorm(num_channels=3)    
    x = fluid.dygraph.to_variable(data)
    y = bn(x)
    print('output of BatchNorm Layer: \n {}'.format(y.numpy()))

# 使用Numpy计算均值、方差和归一化的输出
# 这里对第0个特征进行验证
a = np.array([1,4,7])
a_mean = a.mean()
a_std = a.std()
b = (a - a_mean) / a_std
print('std {}, mean {}, \n output {}'.format(a_mean, a_std, b))

# 建议读者对第1和第2个特征进行验证，观察numpy计算结果与paddle计算结果是否一致
```

```
# 输入数据形状是[N, C, H, W]时的batchnorm示例
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import BatchNorm

# 设置随机数种子，这样可以保证每次运行结果一致
np.random.seed(100)
# 创建数据
data = np.random.rand(2,3,3,3).astype('float32')
# 使用BatchNorm计算归一化的输出
with fluid.dygraph.guard():
    # 输入数据维度[N, C, H, W]，num_channels等于C
    bn = BatchNorm(num_channels=3)
    x = fluid.dygraph.to_variable(data)
    y = bn(x)
    print('input of BatchNorm Layer: \n {}'.format(x.numpy()))
    print('output of BatchNorm Layer: \n {}'.format(y.numpy()))

# 取出data中第0通道的数据，
# 使用numpy计算均值、方差及归一化的输出
a = data[:, 0, :, :]
a_mean = a.mean()
a_std = a.std()
b = (a - a_mean) / a_std
print('channel 0 of input data: \n {}'.format(a))
print('std {}, mean {}, \n output: \n {}'.format(a_mean, a_std, b))

# 提示：这里通过numpy计算出来的输出
# 与BatchNorm算子的结果略有差别，
# 因为在BatchNorm算子为了保证数值的稳定性，
# 在分母里面加上了一个比较小的浮点数epsilon=1e-05
```

#### 预测时使用BatchNorm

上面介绍了在训练过程中使用BatchNorm对一批样本进行归一化的方法，但如果使用同样的方法对需要预测的一批样本进行归一化，则预测结果会出现不确定性。在BatchNorm的具体实现中，训练时会计算均值和方差的移动平均值。


# 丢弃法 Dropout

丢弃法（Dropout）是深度学习中一种常用的抑制过拟合的方法，其做法是在神经网络学习过程中，随机删除一部分神经元。训练时，随机选出一部分神经元，将其输出设置为0，这些神经元将不对外传递信号。

在预测场景时，会向前传递所有神经元的信号，可能会引出一个新的问题：训练时由于部分神经元被随机丢弃了，输出数据的总大小会变小。可采用以下两种方法：

- **downgrade_in_infer**：训练时以比例 r 随机丢弃一部分神经元，不向后传递它们的信号；预测时向后传递所有神经元的信号，但是将每个神经元上的数值乘以 (1−r)。
- **upscale_in_train**：训练时以比例 r 随机丢弃一部分神经元，不向后传递它们的信号，但是将那些被保留的神经元上的数值除以 (1−r)；预测时向后传递所有神经元的信号，不做任何处理。

paddle默认为前者。其 dropout API 包含的主要参数如下：

- x （Variable）- 数据类型是Tensor，需要采用丢弃法进行操作的对象。
- dropout_prob (float32) - 对 x 中元素进行丢弃的概率，即输入单元设置为0的概率，该参数对元素的丢弃概率是针对于每一个元素而言而不是对所有的元素而言。举例说，假设矩阵内有12个数字，经过概率为0.5的dropout未必一定有6个零。
- is_test (bool) - 是否运行在测试阶段，由于dropout在训练和测试阶段表现不一样，通过此参数控制其表现，默认值为'False'。
- dropout_implementation (str) - 丢弃法的实现方式，有'downgrade_in_infer'和'upscale_in_train'两种，具体情况请见上面的说明，默认是'downgrade_in_infer'。


下面这段程序展示了经过dropout之后输出数据的形式。

```
# dropout操作
import numpy as np

import paddle
import paddle.fluid as fluid

# 设置随机数种子，这样可以保证每次运行结果一致
np.random.seed(100)
# 创建数据[N, C, H, W]，一般对应卷积层的输出
data1 = np.random.rand(2,3,3,3).astype('float32')
# 创建数据[N, K]，一般对应全连接层的输出
data2 = np.arange(1,13).reshape([-1, 3]).astype('float32')
# 使用dropout作用在输入数据上
with fluid.dygraph.guard():
    x1 = fluid.dygraph.to_variable(data1)
    out1_1 = fluid.layers.dropout(x1, dropout_prob=0.5, is_test=False)
    out1_2 = fluid.layers.dropout(x1, dropout_prob=0.5, is_test=True)

    x2 = fluid.dygraph.to_variable(data2)
    out2_1 = fluid.layers.dropout(x2, dropout_prob=0.5, \
                    dropout_implementation='upscale_in_train')
    out2_2 = fluid.layers.dropout(x2, dropout_prob=0.5, \
                    dropout_implementation='upscale_in_train', is_test=True)
    
    print('x1 {}, \n out1_1 \n {}, \n out1_2 \n {}'.format(data1, out1_1.numpy(),  out1_2.numpy()))
    print('x2 {}, \n out2_1 \n {}, \n out2_2 \n {}'.format(data2, out2_1.numpy(),  out2_2.numpy()))
```





