# 批次标准化与卷积神经网络 (Batch Normalization & CNN) 学习报告

**视频来源：**
1. [類神經網路訓練不起來怎麼辦(五)： 批次標準化 (Batch Normalization)](https://www.youtube.com/watch?v=BABPWOkSbLE)
2. [卷積神經網路 (Convolutional Neural Networks, CNN)](https://www.youtube.com/watch?v=OP5HcXJg2Aw)

**参考课件：** `normalization_v4.pdf`, `cnn_v4.pdf`

---

## 1. 批次标准化 (Batch Normalization)

### 1.1 为什么需要标准化？(Changing Landscape)
当输入数据的不同维度（特征）的值域范围差异巨大时，Error Surface（误差曲面）会变得非常不规则（比如在一个方向上非常平缓，在另一个方向上非常陡峭）。这会导致我们在使用梯度下降法时难以选择合适的学习率。
**特征标准化 (Feature Normalization)** 通过让所有维度的均值为 0，方差为 1，可以使得误差曲面变得更加平滑 (smooth)，从而让梯度下降收敛得更快。

### 1.2 深度学习中的 Batch Normalization
在深度神经网络中，每一层的输出都会作为下一层的输入。即使我们在最开始对输入 $x$ 做了标准化，经过第一层的权重 $W^1$ 乘加后，得到的值 $z^1$ 其分布又会变得不均匀，这同样会导致后续层难以优化。
因此，**我们需要对每一层（或某些层）的激活前的值 $z$ 也进行标准化**。
*   **计算方法：** 在训练时，我们每次从数据集中取出一个 Batch。对于这个 Batch 内的数据，计算均值 $\mu$ 和标准差 $\sigma$，然后将 $z$ 标准化为 $\tilde{z} = \frac{z - \mu}{\sigma}$。接着，我们通常还会加上两个可学习的参数 $\gamma$ 和 $\beta$：$\hat{z} = \gamma \odot \tilde{z} + \beta$，以保留网络的表达能力。
*   **测试阶段 (Testing)：** 在测试时，我们可能没有一个完整的 Batch 来计算均值和标准差。因此，在训练过程中，我们会计算 $\mu$ 和 $\sigma$ 的**滑动平均值 (Moving Average)**，并在测试时直接使用这些滑动平均值。

### 1.3 为什么 Batch Norm 有效？
早期有一种观点认为 Batch Norm 解决了“内部协变量偏移 (Internal Covariate Shift)”的问题，但后来的实验证明并非如此。真正的原因是，**Batch Normalization 改变了 Error Surface 的地形，使其变得更加平滑，从而让优化变得更加容易。**

---

## 2. 卷积神经网络 (CNN)

CNN 是一种专门为图像处理设计的网络架构。如果直接用全连接网络 (Fully Connected Network) 处理图像（比如 100x100x3 的图片），参数量会变得极其庞大（例如 3000万个参数），不仅计算困难，还容易过拟合。

CNN 的设计基于人类观察图像的三个重要特性 (Observations)：

### 2.1 特性一：不需要看整张图 (Receptive Field)
*   **观察：** 要识别图片中是否有鸟，神经元不需要看整张图片，只需要看到鸟嘴、鸟爪等局部特征即可。
*   **简化：** 我们限制每个神经元只看图像的一个小区域，这个区域称为 **感受野 (Receptive Field)**（例如 3x3x3 的区域）。

### 2.2 特性二：同样的特征会出现在不同位置 (Parameter Sharing)
*   **观察：** 鸟嘴可能出现在图片的左上角，也可能出现在右下角。
*   **简化：** 让负责检测左上角鸟嘴的神经元，和负责检测右下角鸟嘴的神经元 **共享参数 (Parameter Sharing)**。
*   **Filter (滤波器)：** 这两步简化结合起来，就构成了 CNN 中的 **Filter**。每个 Filter 就像是一个特定模式（如鸟嘴）的检测器，它在整张图片上滑动（卷积 Convolution），以此来检测图片各个位置是否出现了该模式。

### 2.3 特性三：下采样不改变物体本质 (Pooling)
*   **观察：** 把一张高分辨率图片的像素去掉一部分（缩小图片），人类依然能认出图片里的物体。
*   **简化：** 引入 **池化层 (Pooling)**，比如 Max Pooling。它将相邻的几个像素合并为一个像素（例如选出最大值），从而成倍地减小图像的尺寸，大幅减少后续计算的参数量。

### 2.4 CNN 的典型架构
一个典型的 CNN 架构是：**[ Convolution -> Pooling ] 交替重复多次 -> Flatten (展平) -> Fully Connected Layers (全连接层) -> Softmax (输出分类结果)**。

**特例分析：AlphaGo 为什么用 CNN？**
围棋棋盘可以看作是一个 19x19 的图像。它同样具有 CNN 的前两个特性（局部模式决定吃子、同样的吃子模式会出现在不同位置），所以 AlphaGo 使用了 CNN 并且表现极好。**但是，AlphaGo 没有使用 Pooling 层！** 因为在围棋中，丢掉任意一行或一列的交叉点（下采样），整个棋局的意义就完全改变了，这与图像缩小依然是原图的特性截然不同。

---

## 总结
*   **Batch Normalization** 通过在网络内部强制规范化数据的分布，平滑了误差曲面，是加速深层网络训练的利器。
*   **CNN** 通过**感受野 (Receptive Field)**、**参数共享 (Parameter Sharing)** 和 **池化 (Pooling)** 三大机制，极大地减少了处理图像时的参数量，引入了针对图像任务量身定制的 Model Bias，是目前计算机视觉领域的基础架构。