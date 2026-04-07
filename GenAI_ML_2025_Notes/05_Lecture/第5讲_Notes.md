# 生成式 AI 与机器学习导论 (2025) - 第5讲 学习笔记

## 执行摘要 (Executive Summary)
本讲为机器学习与深度学习的基础观念入门。李宏毅老师跳脱语言模型的框架，以「预测上课时长」这项生活化的回归任务 (Regression) 为例，具体展示如何「透过资料寻找一个函式 (Function)」，这正是机器学习的核心本质。课程详细拆解了机器学习的「3+1 步骤」：定义损失函数 (Loss)、划定模型范围 (Model)、进行最佳化 (Optimization)，以及不可或缺的验证 (Validation)。此外，课程从简单的线性模型逐步推导至由 ReLU 构成的分段线性曲线，进而带出类神经网路 (Neural Network) 与深度学习 (Deep Learning) 的基本架构，并透过实际的 Python 程式码演练了梯度下降法 (Gradient Descent) 以及应对过拟合 (Overfitting) 的实务技巧。

---

## 核心概念 (Key Concepts)
*   **机器学习 (Machine Learning)**：透过资料自动找出一个未知函式 $f(x)$ 的技术。
*   **回归 (Regression)**：输入资料，输出一个连续数值（例如预测上课时间长度）的任务类型。
*   **特征 (Feature)**：将真实世界的物件（如投影片）转换为能输入给函式的数值（如页数、平均字数）。
*   **参数 (Parameter)**：函式中未知的数值（如权重 $W$、偏置 $b$），需要透过资料学习而来。
*   **模型 (Model)**：人类凭借领域知识 (Domain Knowledge) 所划定的一组候选函式集合。
*   **损失函数 (Loss / Cost)**：评估一个函式好坏的指标。数值越小，代表函式预测越精准。常见指标如均方误差 (Mean Squared Error, MSE)。
*   **梯度下降法 (Gradient Descent)**：用于寻找让 Loss 最小的参数组合的最佳化演算法。透过计算参数对 Loss 的偏微分 (Gradient)，朝让 Loss 下降的方向更新参数。
*   **超参数 (Hyperparameter)**：无法透过机器学习自动找出，必须由人类手动设定的参数，例如学习率 (Learning Rate)、批次大小 (Batch Size)、训练周期 (Epoch)。
*   **类神经网路 (Neural Network) 与 深度学习 (Deep Learning)**：将多个神经元 (Neuron, 即 $W x+b$ 后通过激励函数) 排列成多层 (Hidden Layers) 形成的庞大复杂函式集合，理论上可逼近任何连续函式。
*   **过拟合 (Overfitting)**：模型在训练资料上 Loss 很低，但在未见过的验证或测试资料上 Loss 却很高的现象。通常因为模型选择范围过大或训练过度所致。

---

## 详细解析 (Detailed Breakdown)

### 机器学习寻找函式的「3+1」步骤

#### 步骤一：定义目标 (Define Loss)
*   **目标**：评估一个候选函式的好坏。
*   **作法**：收集**训练资料 (Training Data)**（例如过去上课的投影片页数与实际讲课时长），将资料输入函式得到预测值，再计算预测值与真实答案的差距。
*   **指标**：常用的评估方式为 **Mean Squared Error (MSE)**，将所有预测误差平方后取平均。Loss 越小越好。

#### 步骤二：划定选择范围 (Define Model)
*   **线性模型 (Linear Model)**：最简单的假设，如 $y = w_1 x_1 + b$。这代表我们假设输入 (页数) 与输出 (时间) 呈线性正相关。
*   **非线性模型与分段线性曲线 (Piecewise Linear Curve)**：真实世界通常不是简单的直线。任何复杂的曲线都可以被拆解为多个「分段线性曲线」。
*   **ReLU (Rectified Linear Unit)**：一种山坡状的折线函数 ($max(0, wx+b)$)。将多个 ReLU 组合起来，就可以逼近任何复杂的分段线性曲线。
*   **深度学习架构**：将输入乘以矩阵 $W$ 加上向量 $b$，通过 ReLU 激励函数得到输出 $a$。这一个完整操作称为一个**神经元 (Neuron)** 或一个**隐藏层 (Hidden Layer)**。反复进行多次这种操作叠加，就成了**深度学习 (Deep Learning)**。

#### 步骤三：最佳化寻找最好函式 (Optimization)
*   **目标**：寻找一组参数 $W^*, b^*$，使得 $Loss$ 最小化。
*   **梯度下降法 (Gradient Descent)**：
    1.  随机设定初始参数 $W^0, b^0$。
    2.  计算目前参数对 Loss 的偏微分 (即**梯度 Gradient**，可视为误差曲面的切线斜率)。
    3.  利用公式更新参数：$W^{new} = W^{old} - \text{Learning\_Rate} \times Gradient$。
    4.  反复迭代，直到梯度接近零 (走到谷底) 或达到设定的训练次数。
*   **批次 (Batch) 与 Epoch**：
    *   为了加速计算，不会每次都拿「所有」资料来算 Gradient (Full-Batch)，而是把资料切成小块 (Batch)。
    *   看过所有 Batch 的资料一轮，称为一个 **Epoch**。
    *   在每一个 Epoch 前，通常会将资料打乱 (**Shuffle**)，以增加随机性并帮助模型更好地训练。

#### 步骤「+1」：验证 (Validation)
*   **为何需要验证？** 模型如果在训练资料上表现很好，不代表在未来的实际测试中也表现好。
*   **验证集 (Validation Set)**：切割一部分资料作为「模拟考」，用来挑选最好的模型或调整超参数。
*   **过拟合 (Overfitting) 的解法**：
    *   更换/增加更有意义的特征 (例如：用「平均每页字数」取代「总字数」)。
    *   设定适当的模型复杂度。
    *   **提早停止 (Early Stopping)**：在 Validation Loss 达到最低点时就停止训练，不要硬跑到设定的最大 Epoch 数。
*   **资料集的切分**：实务上与竞赛中，资料会被切分为 Training Set (训练)、Validation Set (验证自己调参)、Public Test Set (排行榜公开测试)、Private Test Set (最终一翻两瞪眼的真实测试)，借此层层防堵 Overfitting。

---

## 结论 (Conclusions)
这堂课生动地揭示了机器学习不是魔法，而是一个「利用数学最佳化方法，从资料中找出规律函式」的严谨工程。
1.  **资料决定上限**：如果训练资料跟实际应用场景差异过大 (如拿2021年的深课程预测2025年的导论课程)，第一步的 Loss 定义就错了，后续训练再好也没有用。
2.  **特征工程 (Feature Engineering) 很重要**：适当的人为知识介入 (把总字数转换为平均字数) 能够大幅降低训练难度并提升预测准确度。
3.  **训练与验证的平衡**：深度学习提供了广大的函式搜寻范围，但极易陷入 Overfitting 或是 Optimization 失败 (例如卡在 Local Minima、Learning Rate 设太大导致 Loss 爆炸等)。掌握如何观察 Training Curve 与 Validation Curve，并适时调整超参数或 Early Stopping，是实务上训练出好模型的必备技能。
4.  **最终验证**：课程最后使用模型预测当天上课时长，成功得出 107 分钟的完美结果，趣味地印证了只要流程与特征正确，机器学习确实能从历史资料中精准预测未来。