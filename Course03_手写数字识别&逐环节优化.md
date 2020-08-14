# MNIST数据集

## 采用房价预测模型的方法

### 数据处理

在手写数字识别任务中，通过``paddle.dataset.mnist``可以直接获取处理好的MNIST训练集、测试集。将``batch_size``设置为8，即一个批次有8个标签。

```
#加载飞桨和相关类库
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
from PIL import Image

# 如果～/.cache/paddle/dataset/mnist/目录下没有MNIST数据，API会自动将MINST数据下载到该文件夹下
# 设置数据读取器，读取MNIST数据训练集
trainset = paddle.dataset.mnist.train()
# 包装数据读取器，每次读取的数据数量设置为batch_size=8
train_reader = paddle.batch(trainset, batch_size=8)
```

通过以下方式读取第一个batch的内容，并查看打印结果：

```
# 以迭代的形式读取数据
for batch_id, data in enumerate(train_reader()):
    # 获得图像数据，并转为float32类型的数组
    img_data = np.array([x[0] for x in data]).astype('float32')
    # 获得图像标签数据，并转为float32类型的数组
    label_data = np.array([x[1] for x in data]).astype('float32')
    # 打印数据形状
    print("图像数据形状和对应数据为:", img_data.shape, img_data[0])
    print("图像标签形状和对应数据为:", label_data.shape, label_data[0])
    break

print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(label_data[0]))
```

(运行结果略）从打印结果看，从数据加载器``train_reader()``中读取一次数据，可以得到形状为（8, 784）的**图像数据**和形状为（8,）的**标签数据**。其中，形状中的数字8与设置的``batch_size``大小对应，784为MINIST数据集中每个图像的像素大小(28*28)。注：飞桨将维度是28*28的手写数字图像转成**向量形式存储**，因此使用飞桨数据加载器读取到的手写数字图像是长度为784（28*28）的**向量**。

此外，从打印的图像数据来看，图像数据的范围是[-1, 1]，表明这是已经完成图像归一化后的图像数据，并且空白背景部分的值是-1。将图像数据反归一化，并使用matplotlib工具包将其显示出来。

```
# 显示第一batch的第一个图像
import matplotlib.pyplot as plt
img = np.array(img_data[0]+1)*127.5
img = np.reshape(img, [28, 28]).astype(np.uint8)

plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()
```

### 模型设计

在房价预测中，使用了单层且线性变换的模型。在手写数字识别中，如果依然使用这个模型，其输入为784维的数据，输出为1维数据。但是，输入**像素的位置排布信息**对理解图像内容非常重要，因此网络的输入设计为28*28的尺寸而不是1*784，以便于模型能够正确处理**像素之间的空间信息**。

(虽然在只有一层的简单网络中，并不能处理位置关系信息，因此该模型的预测效果依然有限。采用**卷积神经网络**则可以解决这个问题。）

```
# 定义mnist数据识别网络结构，同房价预测网络
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        
        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = Linear(input_dim=784, output_dim=1, act=None)
        
    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs
```

### 训练配置

训练配置要先生成模型实例（设为“训练”状态），再设置优化算法（SGD）和学习率（0.001)。

```
# 定义飞桨动态图工作环境
with fluid.dygraph.guard():
    # 声明网络结构
    model = MNIST()
    # 启动训练模式
    model.train()
    # 定义数据读取函数，数据读取batch_size设置为16
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
```

### 训练过程

依旧采用二层循环嵌套方式，训练完成后保存模型参数供后续使用。

```
# 通过with语句创建一个dygraph运行的context
# 动态图下的一些操作需要在guard下进行
with fluid.dygraph.guard():
    model = MNIST()
    model.train()
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，格式需要转换成符合框架要求
            image_data = np.array([x[0] for x in data]).astype('float32')
            label_data = np.array([x[1] for x in data]).astype('float32').reshape(-1, 1)
            # 将数据转为飞桨动态图格式
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程
            predict = model(image)
            
            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.square_error_cost(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id !=0 and batch_id  % 1000 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    # 保存模型
    fluid.save_dygraph(model.state_dict(), 'mnist')
```

### 模型测试

包括以下四步：
- **声明实例**
- **加载模型**：加载训练过程中保存的模型参数。
- **灌入数据**：将测试样本传入模型，模型的状态设置为校验状态（eval），显式告诉框架我们接下来只会使用前向计算的流程，不会计算梯和梯度反向传播。
- 获取预测结果，取整后作为**预测标签输出**。

在模型测试之前，需要先**对样例图片进行归一化**处理。

```
# 导入图像读取第三方库
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
# 读取图像
img1 = cv2.imread('./work/example_0.png')
example = mpimg.imread('./work/example_0.png')
# 显示图像
plt.imshow(example)
plt.show()
im = Image.open('./work/example_0.png').convert('L')
print(np.array(im).shape)
im = im.resize((28, 28), Image.ANTIALIAS)
plt.imshow(im)
plt.show()
print(np.array(im).shape)

# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    print(np.array(im))
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # 图像归一化，保持和数据集的数据范围一致
    im = 1 - im / 127.5
    return im

# 定义预测过程
with fluid.dygraph.guard():
    model = MNIST()
    params_file_path = 'mnist'
    img_path = './work/example_0.png'
# 加载模型参数
    model_dict, _ = fluid.load_dygraph("mnist")
    model.load_dict(model_dict)
# 灌入数据
    model.eval()
    tensor_img = load_image(img_path)
    result = model(fluid.dygraph.to_variable(tensor_img))
#  预测输出取整，即为预测的数字，打印结果
    print("本次预测的数字是", result.numpy().astype('int32'))
```

从打印结果来看，模型预测出的数字是与实际输出的图片的数字不一致。这里只是验证了一个样本的情况，如果我们尝试更多的样本，可发现许多数字图片识别结果是错误的。因此完全复用房价预测的实验并不适用于手写数字识别任务！

## 逐环节优化 - 数据处理

上一节，我们通过调用飞桨提供的API（paddle.dataset.mnist）加载MNIST数据集。但在工业实践中，我们面临的任务和数据环境千差万别，通常需要自己编写适合当前任务的数据处理程序，一般涉及如下五个环节：

- 读入数据
- 划分数据集
- 生成批次数据
- 训练样本集乱序
- 校验数据有效性

首先要加载飞桨和数据处理库
```
# 加载飞桨和相关数据处理的库
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
import gzip
import json
import random
```

### 读入数据并划分数据集

data包含三个元素的列表：train_set、val_set、 test_set。

- **train_set（训练集）**：包含50000条手写数字图片和对应的标签，用于确定模型参数。包含两个元素的列表：train_images、train_labels。
  - **train_images**：[50000, 784]的二维列表，包含50000张图片。每张图片用一个长度为784的向量表示，内容是28*28尺寸的像素灰度值（黑白图片）。
  - **train_labels**：[50000, ]的列表，表示这些图片对应的分类标签，即0-9之间的一个数字。
- **val_set（验证集）**：包含10000条手写数字图片和对应的标签，用于**调节模型超参数**（如多个网络结构、正则化权重的最优选择）。
- **test_set（测试集）**：包含10000条手写数字图片和对应的标签，用于**估计应用效果**（没有在模型中应用过的数据，更贴近模型在真实场景应用的效果）。


#### 扩展阅读

*“拷问数据足够久，它终究会招供”*

假设所有论文共产生1000个模型，这些模型使用的是测试数据集来评判模型效果，并最终选出效果最优的模型。这相当于把原始的测试集当作了验证集，使得测试集失去了真实评判模型效果的能力。

小建议：当几个模型的准确率在测试集上差距不大时，尽量选择网络结构相对简单的模型。往往越精巧设计的模型和方法，越不容易在不同的数据集之间迁移。


### 训练样本乱序、生成批次数据

- **训练样本乱序**： 先将样本按顺序进行编号，建立ID集合index_list。然后将index_list乱序，最后按乱序后的顺序读取数据。通过大量实验发现，**模型对最后出现的数据印象更加深刻**。训练数据导入后，越接近模型训练结束，最后几个批次数据对模型参数的影响越大。为了避免模型记忆影响训练效果，需要进行样本乱序操作。
- **生成批次数据**： 先设置合理的batch_size，再将数据转变成符合模型输入要求的np.array格式返回。同时，在返回数据时将Python生成器设置为``yield``模式，以减少内存占用。关于``yield``的理解可见文章[python中yield的用法详解——最简单，最清晰的解释](https://blog.csdn.net/mieleizhi0522/article/details/82142856)

在执行如上两个操作之前，需要先将数据处理代码封装成``load_data``函数，方便后续调用。``load_data``有三种模型：``train``、``valid``、``eval``，分为对应返回的数据是训练集、验证集、测试集。

```
imgs, labels = train_set[0], train_set[1]
print("训练数据集数量: ", len(imgs))
# 获得数据集长度
imgs_length = len(imgs)
# 定义数据集每个数据的序号，**根据序号读取数据**
index_list = list(range(imgs_length))
# 读入数据时用到的批次大小
BATCHSIZE = 100

# 随机打乱训练数据的索引序号
random.shuffle(index_list)

# 定义数据生成器，返回批次数据
def data_generator():

    imgs_list = []
    labels_list = []
    for i in index_list:
        # 将数据处理成期望的格式，比如类型为float32，shape为[1, 28, 28]
        img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
        label = np.reshape(labels[i], [1]).astype('float32')
        imgs_list.append(img) 
        labels_list.append(label)
        if len(imgs_list) == BATCHSIZE:
            # 获得一个batchsize的数据，并返回
            yield np.array(imgs_list), np.array(labels_list)
            # 清空数据读取列表
            imgs_list = []
            labels_list = []

    # 如果剩余数据的数目小于BATCHSIZE，
    # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
    if len(imgs_list) > 0:
        yield np.array(imgs_list), np.array(labels_list)
    return data_generator
    
# 声明数据读取函数，从训练集中读取数据
train_loader = data_generator
# 以迭代的形式读取数据
for batch_id, data in enumerate(train_loader()):
    image_data, label_data = data
    if batch_id == 0:
        # 打印数据shape和类型
        print("打印第一个batch数据的维度:")
        print("图像维度: {}, 标签维度: {}".format(image_data.shape, label_data.shape))
    break
```

### 校验数据有效性

#### 机器校验

如果数据集中的**图片数量和标签数量不等**，说明数据逻辑存在问题，可使用``assert``语句校验图像数量和标签数据是否一致。

```
    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(label))
```

#### 人工校验

人工校验是指**打印数据输出结果**，观察是否是预期的格式。实现数据处理和加载函数后，我们可以调用它读取一次数据，观察数据的shape和类型是否与函数中设置的一致。

```
# 声明数据读取函数，从训练集中读取数据
train_loader = data_generator
# 以迭代的形式读取数据
for batch_id, data in enumerate(train_loader()):
    image_data, label_data = data
    if batch_id == 0:
        # 打印数据shape和类型
        print("打印第一个batch数据的维度，以及数据的类型:")
        print("图像维度: {}, 标签维度: {}, 图像数据类型: {}, 标签数据类型: {}".format(image_data.shape, label_data.shape, type(image_data), type(label_data)))
    break
```

### 封装数据读取与处理函数

下面将这些步骤放在一个函数中实现，方便在神经网络训练时直接调用。

```
def load_data(mode='train'):
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    # 加载json数据文件
    data = json.load(gzip.open(datafile))
    print('mnist dataset load done')
   
    # 读取到的数据区分训练集，验证集，测试集
    train_set, val_set, eval_set = data
    if mode=='train':
        # 获得训练数据集
        imgs, labels = train_set[0], train_set[1]
    elif mode=='valid':
        # 获得验证数据集
        imgs, labels = val_set[0], val_set[1]
    elif mode=='eval':
        # 获得测试数据集
        imgs, labels = eval_set[0], eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    print("训练数据集数量: ", len(imgs))
    
    # 校验数据
    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(label))
    
    # 获得数据集长度
    imgs_length = len(imgs)
    
    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(imgs_length))
    # 读入数据时用到的批次大小
    BATCHSIZE = 100
    
    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            # 训练模式下打乱数据
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            # 将数据处理成希望的格式，比如类型为float32，shape为[1, 28, 28]
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('float32')
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                # 获得一个batchsize的数据，并返回
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据读取列表
                imgs_list = []
                labels_list = []
    
        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)
    return data_generator
```
下面定义一层神经网络，利用定义好的数据处理函数，完成神经网络的训练。

```
# 数据处理部分之后的代码，数据读取的部分调用load_data函数
# 定义网络结构，同上一节所使用的网络结构
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc = Linear(input_dim=784, output_dim=1, act=None)

    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, (-1, 784))
        outputs = self.fc(inputs)
        return outputs

# 训练配置，并启动训练过程
with fluid.dygraph.guard():
    model = MNIST()
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程
            predict = model(image)
            
            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.square_error_cost(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    #保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')
```

=====================================

### 异步数据读取

对于样本量较大、数据读取较慢的场景，建议采用异步数据读取方式。异步读取数据时，数据读取和模型训练并行执行，从而加快了数据读取速度，牺牲一小部分内存换取数据读取效率的提升。

- **同步数据读取**：数据读取与模型训练串行。当模型需要数据时，才运行数据读取函数获得当前批次的数据。在读取数据期间，模型一直等待数据读取结束才进行训练，数据读取速度相对较慢。
- **异步数据读取**：数据读取和模型训练并行。读取到的数据不断的放入缓存区，无需等待模型训练就可以启动下一轮数据读取。当模型训练完一个批次后，不用等待数据读取过程，直接从缓存区获得下一批次数据进行训练，从而加快了数据读取速度。
- **异步队列**：数据读取和模型训练交互的仓库，二者均可以从仓库中读取数据，它的存在使得两者的工作节奏可以解耦。

异步数据读取的飞桨实现：
```

# 定义数据读取后存放的位置，CPU或者GPU，这里使用CPU
# place = fluid.CUDAPlace(0) 时，数据才读取到GPU上
place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    # 声明数据加载函数，使用训练模式
    train_loader = load_data(mode='train')
    # 定义DataLoader对象用于加载Python生成器产生的数据
    data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
    # 设置数据生成器
    data_loader.set_batch_generator(train_loader, places=place)
    # 迭代的读取数据并打印数据的形状
    for i, data in enumerate(data_loader):
        image_data, label_data = data
        print(i, image_data.shape, label_data.shape)
        if i>=5:
            break
```

fluid.io.DataLoader.from_generator参数名称和含义如下：

- feed_list：仅在PaddlePaddle静态图中使用，动态图中设置为“None”，本教程默认使用动态图的建模方式；
- capacity：表示在DataLoader中维护的队列容量，如果读取数据的速度很快，建议设置为更大的值；
- use_double_buffer：是一个布尔型的参数，设置为“True”时，Dataloader会预先异步读取下一个batch的数据并放到缓存区；
- iterable：表示创建的Dataloader对象是否是可迭代的，一般设置为“True”；
- return_list：在动态图模式下需要设置为“True”。




异步读取数据只在数据量规模巨大时会带来显著的性能提升，对于多数场景采用同步数据读取的方式已经足够。












