# 生成式 AI 与机器学习 2025 - 第三讲学习笔记：解剖语言模型内部运作

## 一、 执行摘要 (Executive Summary)
本讲次深入探讨了大型语言模型（LLM）的内部运作机制，将语言模型视为一个数学函式 $F(X)$，在不涉及模型训练的前提下，直接「解剖」已训练好的模型（如 Llama 3B 与 Gemma 4B）。课程从输入文字如何转换为机率分布的宏观视角出发，依序解析 Tokenization、Embedding Table、多层 Transformer 网路、以及最后的 LM Head 与 Softmax 机制。随后，课程介绍了几种分析模型内部隐藏表征（Hidden Representations）的前沿技术，包括表征工程（Representation Engineering）、Logit Lens 与 Patchscopes。最后，课程微观剖析了 Transformer 单一层的内部结构，详细讲解了自注意力机制（Self-Attention）与前馈神经网路（Feed-Forward Network）的数学原理与物理意义，并透过实际的程式码操作验证了这些理论。

## 二、 核心概念 (Key Concepts)
1. **Token 与 Embedding Table**：输入的句子被切分为 Token（ID），透过 Embedding Table（一个巨大的权重矩阵）转换为具有语意空间分布的密集向量（Token Embedding）。
2. **多层网路架构 (Deep Learning)**：模型由多个 Layer 组成，逐层将输入的 Token Embedding 转换为考虑了上下文的 Contextualized Embedding（或称 Hidden/Latent Representation）。
3. **LM Head 与 Softmax**：最后一层的输出向量会乘上 LM Head 矩阵（通常与 Embedding Table 共用参数，称为 Unembedding），得到 Logit。Logit 接著透过 Softmax 转换为介于 0~1 的机率分布，并可透过温度参数（Temperature, $T$）控制生成的创造力或保守程度。
4. **自注意力机制 (Self-Attention)**：Transformer 的核心，透过 Query ($Q$)、Key ($K$)、Value ($V$) 矩阵的运算，计算 Token 之间的关联度（Attention Weights），并对 Value 进行加权总和（Weighted Sum）来融合上下文资讯。
5. **多头注意力 (Multi-head Attention) 与因果注意力 (Causal Attention)**：拥有多组 $Q, K, V$ 以捕捉不同面向的上下文关系；在生成式任务中，每个 Token 只能 attend 到其左侧（过去）的 Token。
6. **表征工程 (Representation Engineering)**：透过人为提取并加减特定的表征向量（例如「拒绝请求」的成分或「谄媚」的成分），可直接操控模型的行为，而无需重新训练。
7. **Logit Lens 与 Patchscopes**：窥探语言模型逐层「思考」过程的分析技术，透过提早将中间层的表征映射回词汇空间，了解模型在生成每个 Token 前的内心变化。

## 三、 详细解析 (Detailed Breakdown)

### 3.1 语言模型从输入到输出的完整流程
*   **Tokenization**：将输入提示（Prompt）切分为 ID 序列。不同模型的 Token 词表大小不同（Llama 3B 约 12.8 万，Gemma 4B 约 26.2 万）。
*   **Token Embedding**：查表操作，将每个 ID 映射为一个高维度的向量（例如 3072 维或 2560 维）。意思相近的 Token，其 Embedding 在高维空间中的距离也会较近（例如 "apple" 与 "iPhone"、"MacBook" 相关）。
*   **Transformer Layers**：向量依序通过 $L$ 个 Layer。每一层的输入与输出长度相同，且会综合考虑前面的 Token，产生 Contextualized Embedding。
*   **LM Head (Unembedding)**：将最后一层的输出向量乘上一个形状为 `(Vocabulary Size) x (Dimension)` 的矩阵，计算该向量与所有 Token Embedding 的内积（相似度），得到长度为 Vocabulary Size 的向量（Logits）。
*   **Softmax 与 Temperature**：Logits 透过 Softmax 转换为机率。公式包含除以温度 $T$：
    *   $T$ 越大：机率分布越平缓，容易生成罕见词汇（创意模式）。
    *   $T$ 越小：机率分布越集中，倾向生成机率最高的词汇（保守模式）。

### 3.2 窥探与操控语言模型的内部表征
*   **分析 Representation 的变化**：以 "apple" 为例，当它代表「可食用的苹果」与代表「苹果电脑」时，在第 0 层的 Token Embedding 是一模一样的；但随著层数加深，模型吸收了上下文资讯，两者的 Representation 会越来越不同；相反地，同样代表「苹果电脑」的 "apple"，即使出现在不同的上下文中，其深层 Representation 的相似度依然很高。
*   **Representation Engineering**：将模型在处理「被要求做坏事而拒绝」的隐藏向量，减去「被要求做正常事而答应」的隐藏向量，可以提取出一个纯粹代表「拒绝」的向量。将此向量人为地加到其他正常输入的中间层中，语言模型就会莫名其妙地拒绝正常请求（反之亦然，扣除拒绝向量就能绕过安全限制，或是加入「谄媚」向量让模型疯狂拍马屁）。
*   **Logit Lens**：将未到最后一层的中间层 Representation，提早送入 LM Head 进行 Unembedding，借此观察模型在思考的过渡阶段「心里正准备说什么字」。
*   **Patchscopes**：将某个复杂输入（如「Diana, Princess of Wales」）在中间层产生的 Representation，拼接到另一个模板句子（如「请简单介绍 X」）的中间层中，让模型将其解码成人类可读的描述，借此发现模型在浅层只看到 "Wales" (国家)，中层看到 "Princess" (王室成员)，到深层才真正理解这是 "Diana" (黛安娜王妃) 本人。

### 3.3 单层 Transformer 内部运作机制
*   **Self-Attention Layer**：
    *   负责处理上下文资讯融合。在 2017 年的论文 "Attention is all you need" 中被提出，取代了不易平行化运算的 RNN / LSTM 结构。
    *   **运算步骤**：
        1. **Query ($Q$) 与 Key ($K$)**：当前 Token 乘上 $W^Q$ 得到 $Q$，其他 Token 乘上 $W^K$ 得到 $K$。$Q$ 与所有 $K$ 做点积 (Dot Product)，计算出**注意力权重 (Attention Weight)**，代表其他 Token 对当前 Token 的影响力。
        2. **Positional Embedding**：为了解决单纯点积无法区分词汇位置的问题，必须加入位置编码（如现代模型常用的 RoPE 技术）。
        3. **Value ($V$) 与 Weighted Sum**：各 Token 乘上 $W^V$ 得到 $V$。将所有 $V$ 依据前述算出的 Attention Weight 进行加权总和，得到融合了上下文的新向量。
        4. **Residual Connection (残差连接)**：将新算出的向量加上原本的 Token Embedding，作为该模组的输出。
    *   **Multi-head Attention**：拥有多组不同的 $W^Q, W^K, W^V$。例如，某些 Head 专门寻找形容词，某些 Head 专门寻找数量词。所有 Head 的结果最后透过矩阵 $W^O$ 拼接融合。
*   **Feed-Forward Network (FFN)**：
    *   Attention Layer 的输出会再通过全连接层。
    *   操作包含：矩阵相乘、加上 Bias、通过 Activation Function (如 ReLU, GeLU)、再进行一次矩阵相乘与加 Bias。
    *   学术上有将 FFN 解读为 Key-Value Memories 的观点，即它是另一种维度的注意力机制。
    *   所谓的「神经元 (Neuron)」与「神经网路」，本质上就只是这类包含启动函数在内的连续矩阵与向量运算，为了向大众解释而包装的生动名词。

### 3.4 实作观察 (Llama 3B vs Gemma 4B)
*   **参数规模**：Llama 3B 拥有约 32 亿参数（28 层），而 Gemma 4B 拥有约 43 亿参数（34 层，包含 Vision 处理能力）。
*   **Grouped-Query Attention (GQA)**：在检视 Llama 的 Q, K, V 矩阵形状时发现，Q 矩阵输出是 3072 维，但 K 和 V 输出只有 1024 维，这是一种为了节省参数与运算量而设计的特殊 Attention 架构。
*   **Attention 视觉化**：绘制两两 Token 之间的 Attention Weights 矩阵时，发现右上角全为 0（因 Causal Attention 的限制），且许多 Attention Head 倾向于关注句子的「起始符号」（Start Token）。这意味著当该 Head 在当前语境中找不到需要关注的对象时，会将起始符号作为预设的「垃圾桶」来分配权重。

## 四、 结论 (Conclusions)
语言模型的「智慧」并非黑魔法，而是建立在巨量参数与连续矩阵运算之上的数学模型。透过拆解 Transformer 的内部架构，我们理解了 Token 如何透过 Embedding Table 转换为具备语意空间分布的向量，并在逐层的 Self-Attention 与 Feed-Forward Network 中逐步叠加、融合上下文资讯，最终转化为预测下一个 Token 的机率分布。

本讲次引入的各种探针与操控技术（Logit Lens、Patchscopes、Representation Engineering）进一步揭开了 LLM 处理资讯的黑盒子。这些研究证实，语言模型的内部隐藏状态确实具备可解释性与可控性。透过理解这些内部运作机制，开发者不仅能更清晰地解读模型产生的结果，未来更有机会从根本（例如直接介入表征向量）上去修正模型的偏见、幻觉，或控制其安全性与对齐度（Alignment）。
