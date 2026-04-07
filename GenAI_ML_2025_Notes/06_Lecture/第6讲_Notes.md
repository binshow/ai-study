# 第6讲 机器学习笔记：训练类神经网路的实用技巧

## 执行摘要 (Executive Summary)
本讲次深入探讨了在训练类神经网路时常用的各种关键技术与小技巧。课程的核心在于建立一个评估新技术的框架：当我们遇到一个新方法时，必须思考它改变了机器学习三步骤（定义损失函数、设定函式范围、最佳化寻找最佳函式）中的哪一步，以及它带来的是「**更好的优化 (Optimization)**」还是「**更好的泛化 (Generalization)**」。透过这个框架，我们能针对模型训练时遇到的具体问题（如训练误差降不下来或发生过拟合），对症下药选择合适的技术。

---

## 核心概念 (Key Concepts)
1. **Optimization vs. Generalization (优化与泛化)**
   - **Optimization 问题**：模型的 Training Loss 降不下去，甚至不如简单的线性模型。这通常代表模型卡在局部最小值 (Local Minima) 或鞍点 (Saddle Point)，或是梯度过小。
   - **Overfitting (过拟合) / Generalization 问题**：Training Loss 很低，但 Validation Loss 很高，代表模型在未看过的资料上泛化能力差。
2. **优化技巧 (Optimization Techniques)**
   - Momentum (动量)
   - Adaptive Learning Rate (Adagrad, RMSprop, Adam)
   - Learning Rate Scheduling (Warm-up & Decay)
   - Skip Connection (残差连接)
   - Normalization (正规化，如 Batch Normalization)
3. **泛化技巧 (Generalization Techniques)**
   - CNN (卷积神经网路)
   - Data Augmentation (资料扩增，如 Mixup)
   - Semi-supervised Learning (半监督学习中的 Entropy Minimization)
4. **分类问题与损失函数 (Classification & Loss)**
   - 生成式 AI 的本质也是一系列的分类问题。分类任务通常使用 Cross-Entropy 作为损失函数。

---

## 详细内容解析 (Detailed Breakdown)

### 第一部分：机器学习的挑战与技术分类框架
机器学习的三大步骤为：1. 定义 Loss Function、2. 决定函式选择范围 (Network Architecture)、3. 优化寻找最佳函式 (Optimization)。
当训练深层网路时，如果 Training Loss 降得不如 Linear Model 低，这并非 Overfitting，而是 Optimization 出了问题；只有在 Training Loss 够低但 Validation Loss 居高不下时，才是 Overfitting。学习任何新技术时，都应先问自己：这个技术解决的是 Optimization 问题还是 Generalization 问题？

### 第二部分：优化技巧 (Optimization Techniques)
这些技巧的目标是让模型在训练集上能找到更低的 Loss：
*   **动量 (Momentum)**：
    *   **问题**：在优化过程中，不仅仅是 Local Minima 会让训练停止，Saddle Point (鞍点) 或平坦区域 (梯度极小但非零) 也会使梯度下降法停滞。
    *   **解法**：借鉴物理学的动量概念，将过去所有的梯度计算进行加总与平均。即使当下的梯度为零，模型仍能依赖过去累积的「动量」继续更新参数，有机会越过鞍点或小丘陵。
*   **自适应学习率 (Adaptive Learning Rate)**：
    *   **Adagrad**：根据过去梯度的平方和来调整学习率，但对近期梯度的反应较慢。
    *   **RMSprop**：给予近期梯度较大的权重 (透过参数 $\alpha$)，能更即时地反映 Loss Surface 的地形变化。
    *   **Adam**：结合了 Momentum (考虑方向) 与 RMSprop (考虑大小) 的优点，是现今最常用且强大的优化器。
*   **学习率排程 (Learning Rate Scheduling)**：
    *   **Warm-up**：在训练初期，将学习率从小逐渐调大，让优化器有机会大范围「探索」地形，收集更准确的梯度统计资讯。
    *   **Decay**：训练后期，将学习率逐渐调小，帮助模型平稳著陆，收敛在峡谷的最低点。
*   **残差连接 (Skip Connection / Residual Connection)**：
    *   **好处**：透过在网路层之间加入「高速公路」，可以有效缓解梯度消失 (Gradient Vanishing) 或梯度爆炸 (Gradient Explode) 的问题。
    *   **原理**：它改变了网路架构，使低层的神经元能直接对最终输出产生影响，从而让 Loss Surface 变得更加平坦，大幅降低优化的难度。
*   **正规化 (Normalization)**：
    *   例如 Batch Normalization 或 Layer Normalization。
    *   将每一层的输出强制限制在特定范围内 (如 Mean=0, Variance=1)，使不同维度的数值范围一致，从而更容易调整学习率，显著提升优化效率。同时对泛化也有微小的帮助。

### 第三部分：泛化技巧 (Generalization Techniques)
这些技巧的目标是拉近 Training Loss 与 Validation Loss 的差距，避免过拟合：
*   **卷积神经网路 (CNN)**：
    *   **原理**：利用人类对影像的领域知识 (Domain Knowledge)，透过 Receptive Field (感受野) 与 Parameter Sharing (参数共享)，简化了 Fully Connected Layer。
    *   **好处**：CNN 缩小了函数的搜寻范围，但这个范围是一个「好的范围」，因此能有效避免模型在影像任务上发生 Overfitting。
*   **资料扩增 (Data Augmentation)**：
    *   例如将猫与狗的图片各取 50% 叠加在一起的 Mixup 技术。
    *   **特性**：虽然增加了优化的难度 (可能导致 Training Loss 上升)，但能提供模型更多样的数据，显著改善 Generalization。
*   **半监督式学习 (Semi-supervised Learning)**：
    *   在有标注数据稀缺的情况下，利用大量无标注数据 (Unlabeled Data) 辅助训练。
    *   **Entropy Minimization (熵最小化)**：假设资料是「非黑即白」的。对于无标注图片，虽然不知道正确类别，但我们要求模型输出的机率分布的 Entropy 越小越好 (即越确定越好)，借此拉开决策边界，提升泛化能力。

### 第四部分：分类问题与损失函数 (Classification & Loss)
*   **分类 vs 回归**：上一讲主要讨论回归问题 (使用 MSE)。对于分类问题 (如影像分类或生成式 AI 的文字接龙)，输出通常需经过处理形成机率分布。
*   **交叉熵 (Cross-Entropy)**：在分类任务中，我们使用 Cross-Entropy 作为 Loss Function，来衡量模型输出分布与真实标签 (Ground Truth) 分布之间的差距。

---

## 结论 (Conclusions)
训练深度学习模型并非盲目套用所有最新技术，而是需要具备「诊断问题」的能力。
- 当遇到 **Training Loss 降不下去** 时，应优先考虑替换 Optimizer (如 Adam)、调整 Learning Rate Scheduling (加入 Warm-up)、或是修改模型架构 (加入 Skip Connection 或 Normalization)。
- 当遇到 **Training Loss 很低但 Validation Loss 很高 (Overfitting)** 时，则应该从限制模型复杂度著手，例如引入领域知识修改架构 (如影像任务改用 CNN)、使用 Data Augmentation 增加资料多样性，或善用 Unlabeled Data 进行半监督学习。
灵活运用这个分类框架，才能在实际应用中找出最适合的解决方案，训练出强大且具备高泛化能力的机器学习模型。