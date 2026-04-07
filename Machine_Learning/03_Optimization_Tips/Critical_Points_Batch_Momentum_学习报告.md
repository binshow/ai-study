# 神经网络训练不起来怎么办：临界点、批次与动量 (Optimization Tips)

**视频来源：** 
1. [類神經網路訓練不起來怎麼辦(一)： 局部最小值 (local minima) 與鞍點 (saddle point)](https://www.youtube.com/watch?v=QW6uINn7uGk)
2. [類神經網路訓練不起來怎麼辦(二)： 批次(batch) 與動量(momentum)](https://www.youtube.com/watch?v=zzbr1h9sF54)

**参考课件：** `small-gradient-v7.pdf`

---

## 1. 为什么 Optimization 会失败？(When gradient is small...)

在训练神经网络时，如果发现 Loss 降不下去，很多时候是因为参数更新到了一个梯度接近于 0 的地方。梯度为 0 的点在数学上被称为 **临界点 (Critical Point)**。
临界点通常分为两种情况：
*   **局部最小值 (Local Minima):** 四周的 Loss 都比当前位置高，在这个点上“无路可走”。
*   **鞍点 (Saddle Point):** 形状像马鞍，虽然当前梯度为 0，但在某些方向上 Loss 会上升，在某些方向上 Loss 会下降。如果是鞍点，我们是有办法“逃离”的。

### 如何分辨 Local Minima 和 Saddle Point？
在数学上，我们可以通过泰勒展开式 (Taylor Series Approximation) 引入二阶导数矩阵——**海森矩阵 (Hessian Matrix, $H$)** 来判断：
*   如果 $H$ 的所有特征值 (Eigen values) 都是正数 $\rightarrow$ 这是一个 **Local Minima**。
*   如果 $H$ 的所有特征值都是负数 $\rightarrow$ 这是一个 **Local Maxima**。
*   如果 $H$ 的特征值有正有负 $\rightarrow$ 这是一个 **Saddle Point**。

**好消息是：** 如果是鞍点，我们可以沿着海森矩阵中负特征值对应的特征向量 (Eigenvector) 方向更新参数，从而逃离鞍点并使 Loss 继续下降。
**更高维度的视角：** 在三维空间中看似无路可走的 Local Minima，在极高维度的神经网络参数空间中，往往存在着可以下降的路径。经验表明，**真正的 Local Minima 是非常罕见的**，我们遇到的大多数梯度为 0 的点其实都是鞍点。

---

## 2. 批次 (Batch Size) 的选择：大批次 vs. 小批次

在实际训练中，我们通常不会一次性把所有数据喂给模型，而是将数据分成若干个 **批次 (Batch)** 进行迭代。

### 大批次 (Large Batch / Full Batch)
*   **优点：** 梯度的计算非常稳定准确。得益于现代 GPU 的并行计算能力，如果 Batch 不至于大得撑爆显存，跑完一个 Epoch 的时间反而比小批次更快。
*   **缺点：** 容易卡在鞍点或陷入 **尖锐的最小值 (Sharp Minima)**。在 Sharp Minima 处，训练集上的表现虽好，但稍微遇到一点数据偏移（测试集），Loss 就会急剧上升，导致**泛化能力 (Generalization) 差**。

### 小批次 (Small Batch)
*   **优点：** 虽然每次更新梯度的方向带有很强的“随机噪声 (Noisy)”，但这种噪声反而帮助模型轻易地“震荡”出鞍点或较浅的坑。小批次更容易引导模型走到 **平缓的最小值 (Flat Minima)**，这使得模型在面对未知的测试集时表现更加稳定和优秀。
*   **缺点：** 无法充分利用 GPU 并行算力，导致跑完一个 Epoch 的时间较长。

**结论：** 较小的 Batch Size 往往能在测试集上获得更好的性能（Generalization Better），它是训练时一个非常重要的超参数。

---

## 3. 动量 (Momentum)：物理世界的启示

为了解决模型卡在鞍点或平缓高原 (Plateau) 的问题，研究者从物理世界中获得了灵感：**惯性**。

想象一颗球从山上滚下来，当它滚到一个小坑（局部最小值）或平地（梯度为 0 的鞍点）时，因为物理上的惯性，它大概率会继续向前冲，甚至冲出小坑。

在梯度下降中加入动量 (Gradient Descent + Momentum)：
*   **传统的梯度下降：** 下一步的移动方向 (Movement) = 负的当前梯度。
*   **引入 Momentum：** 下一步的移动方向 = **负的当前梯度 + 上一步的移动方向 (Last Movement)**。

通过引入动量，每一次的参数更新都包含了历史梯度的加权累加。即便当前位置梯度为 0，只要之前积累的“动量”还在，参数就能继续更新，从而非常有效地帮助模型跨越平缓区域，逃离鞍点。

---

## 总结
当你的神经网络训练不起来时，不要轻易认为是模型结构不行，往往是优化过程卡在了临界点。
1. 真正的 Local Minima 很少，大多数是**鞍点**。
2. 使用 **Small Batch Size** 引入噪声，有助于找到泛化能力更强的 Flat Minima。
3. 开启 **Momentum**，利用“惯性”帮助模型冲过鞍点和平缓区域。