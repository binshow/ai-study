# Sequence-to-sequence (Seq2seq) 与 Transformer 学习报告

**视频来源：**
1. [類神經網路訓練不起來怎麼辦(八)： Transformer (上) / Sequence-to-sequence (Seq2seq)](https://www.youtube.com/watch?v=n9TlOhRjYoc)
2. [類神經網路訓練不起來怎麼辦(九)： Transformer (下) / Sequence-to-sequence (Seq2seq)](https://www.youtube.com/watch?v=N6aRv06iv2g)

**参考课件：** `seq2seq_v9.pdf`

---

## 1. 什么是 Seq2seq 模型？

在之前的学习中，我们讨论了输入是一个向量序列，输出是固定类别（或长度与输入相同的序列）的任务。但在很多现实场景中，**不仅输入是一个序列，输出也是一个序列，而且输出的长度是由模型自己决定的**。这种模型就被称为 **Sequence-to-sequence (Seq2seq)** 模型。

### 1.1 Seq2seq 的广泛应用
*   **语音识别 (Speech Recognition):** 输入一段声音（声学特征序列），输出一段文字（字符序列）。
*   **机器翻译 (Machine Translation):** 输入一句英文序列，输出一句长度通常不同的中文序列。
*   **聊天机器人 (Chatbot):** 输入用户的问句，输出机器人的回答。
*   **问答系统 (Question Answering, QA):** 甚至可以把阅读理解的上下文和问题拼接成输入序列，让 Seq2seq 直接输出答案序列。
*   **句法分析 (Syntactic Parsing):** 输入一个句子，输出其对应的树状语法结构（将其转化为括号表示的序列）。
*   **多标签分类 (Multi-label Classification):** 当一个物体可能属于多个类别（且类别数量不固定）时，可以让 Seq2seq 一次输出一个类别，直到模型决定停止。

---

## 2. Transformer 的架构 (基于 Seq2seq)

Seq2seq 模型通常由两部分组成：**Encoder（编码器）** 和 **Decoder（解码器）**。目前 Seq2seq 领域最著名的霸主就是 **Transformer** 模型。

### 2.1 Encoder (编码器)
Encoder 的任务是接收输入序列，并输出一个同样长度但包含更丰富上下文信息的特征序列。
*   **基本结构：** 在 Transformer 中，Encoder 由多个 Block 堆叠而成。
*   **Block 内部构造：**
    1.  首先，输入向量经过 **Multi-Head Self-Attention** 处理，提取全局上下文信息。
    2.  接着引入 **残差连接 (Residual Connection, $a + b$)** 和 **Layer Normalization** (注意：不是 Batch Normalization。Layer Norm 是对同一个样本的不同特征维度做标准化，不需要等待一个 Batch)。
    3.  然后经过一个全连接前馈网络 **(Feed Forward Network, FFN)**。
    4.  最后再做一次残差连接和 Layer Normalization。

### 2.2 Decoder (解码器) - Autoregressive (自回归)
Decoder 的任务是根据 Encoder 提取的特征，**逐字（Token）** 生成输出序列。
以语音识别（输出中文）为例：
1.  首先给 Decoder 输入一个特殊的起始符号 `<START>`（或 `<BOS>`）。
2.  Decoder 结合 Encoder 的信息，输出一个在整个词表（Vocabulary）上的概率分布，选取概率最大的字（如“機”）。
3.  **自回归特性：** 在下一步，将上一步输出的“機”作为 Decoder 的**新输入**，结合上下文继续输出“器”。
4.  这个过程不断循环，直到 Decoder 输出一个特殊的停止符号 `<END>`（或 `<EOS>`），生成过程才宣告结束。这就是为什么 Seq2seq 模型可以决定输出长度的原因。

#### Decoder 的特殊结构：
*   **Masked Multi-Head Self-Attention:** 与 Encoder 不同，Decoder 在生成当前字时，**不能“偷看”未来的字**。因此，计算 Self-Attention 时必须加上 Mask，强制模型只能基于当前位置及之前的 Token 计算 Attention。
*   **Cross Attention (Encoder-Decoder Attention):** 这是连接 Encoder 和 Decoder 的桥梁。Decoder 在生成字时，会拿出自己的 Query ($q$)，去和 Encoder 输出的所有特征向量的 Key ($k$) 和 Value ($v$) 做 Attention 计算，从而提取到源序列中对当前翻译最关键的信息。

---

## 3. Decoder 的进阶机制与训练技巧

### 3.1 Non-autoregressive (NAT) Decoder
前面提到，传统的 Decoder 必须一个个字按顺序生成（AT），无法并行，速度慢。
**NAT (非自回归) Decoder** 试图一次性并行输出整个句子。
*   **方法：** 先用一个额外的预测器估计出输出句子的总长度 $L$，然后直接给 NAT Decoder 输入 $L$ 个 `<START>` 符号，让它一步并行输出 $L$ 个字。
*   **优缺点：** 速度极快，但通常准确率不如 AT Decoder 稳定。在某些任务（如语音合成 TTS）中应用较多。

### 3.2 Teacher Forcing
在训练 Decoder 时，如果第一步模型预测错了（比如应该输出“機”，模型输出了“鬼”），如果把“鬼”当成第二步的输入，会导致一步错、步步错，模型极难收敛。
因此，在训练阶段，我们使用 **Teacher Forcing** 技术：无论 Decoder 上一步输出了什么，我们在下一步都强制把 **Ground Truth（正确答案）** 喂给 Decoder 作为输入。

### 3.3 曝光偏差 (Exposure Bias) & Scheduled Sampling
*   **问题：** 训练时有 Teacher Forcing 保护，Decoder 总是能看到完美的历史输入；但测试时，Decoder 只能靠自己，一旦某一步预测错，就可能产生蝴蝶效应（这叫 Exposure bias）。
*   **解决：** 引入 **Scheduled Sampling**。在训练过程中，不要总是 100% 喂正确答案，偶尔也按一定概率喂模型自己上一轮的输出，让模型学会在犯错的环境下如何纠正自己，从而拉近训练和测试时的环境差异。

### 3.4 复制机制 (Copy Mechanism)
在聊天机器人或文本摘要任务中，有时候模型不需要自己生成词汇，而是直接把输入句子中的某些专有名词（如人名、地名、URL）**复制 (Copy)** 到输出中。像 Pointer Network 就通过 Attention 分数直接决定从输入中复制哪个词。

### 3.5 束搜索 (Beam Search)
在测试生成序列时，每一步总是挑概率最大的词（贪婪解码 Greedy Decoding）最终不一定能得到整体概率最大（最好）的句子。
**Beam Search** 是一种启发式算法，它在每一步保留 Top-K 个最有可能的局部路径（分支），继续往下探索，从而更容易找到全局次优或最优的生成序列。

---

## 总结
Seq2seq 是处理输入序列到输出序列（且长度可变）的终极框架。而以 Self-Attention 为核心构建的 **Transformer** 模型，其 Encoder-Decoder 架构彻底革新了 Seq2seq 的实现方式。
掌握 Encoder 的全局特征提取能力、Decoder 的 Masked Attention 与自回归生成机制，以及 Teacher Forcing、Cross Attention 等关键细节，是理解当今大语言模型（如 GPT 是 Decoder-only，BERT 是 Encoder-only）的必经之路。