# 【生成式人工智慧与机器学习导论2025】第 9 讲：影像和声音上的生成策略

## 执行摘要 (Executive Summary)
本讲探讨了生成式人工智慧在「影像」与「声音」生成上的技术发展与策略。不同于语言模型主要依赖文字接龙 (Autoregressive)，影像与声音的生成面临连续讯号与高维度资料的挑战。课程回顾了从早期的像素/取样点接龙（如 WaveNet、Video Pixel Network），到当前主流的连续标记 (Continuous Token) 与扩散模型 (Diffusion Models) / 流量匹配 (Flow Matching) 的演进，并深入探讨这两条技术路线（接龙与生成模型）如何交会，形成现代强大的多模态生成系统。

## 核心概念 (Key Concepts)
1. **生成式人工智慧的多模态应用：** AI 已经能透过文字指令精准生成包含合理文字的图片（如 NanoBanana）与高品质语音。
2. **像素/取样点接龙 (Pixel/Sample-level Autoregressive)：** 早期尝试将影像的每个像素或声音的每个取样点视为 Token 进行接龙生成，如 WaveNet。虽然效果惊人，但效率极低。
3. **离散标记 (Discrete Token) 与连续标记 (Continuous Token)：** 
   - 为了提高效率，我们需要将语音和影像压缩成更大的基本单位。
   - 使用 Vector Quantization (VQ) 可以将连续讯号转换为离散的 Token 字典，但会损失细节（因为强迫分类）。
   - 连续标记则保留了无限的细微变化，但无法直接应用传统的离散接龙模型，因为其输出会是一整个机率分布的范围。
4. **生成模型 (Generative Models)：**
   - **Diffusion Models & Flow Matching：** 这些模型不依赖简单的 MSE 误差，而是学习如何将一个简单的分布（如常态分布/杂讯）逐步转换为目标的复杂分布（如真实影像的特征）。
5. **两条世界线的交会：** 现代模型将 Autoregressive (接龙) 与 Generative Models 结合。语言模型负责预测下一个 Continuous Token 的条件，而 Generative Model (如 Diffusion) 则负责根据这些条件，从杂讯中生成出精确的连续向量。

## 详细解析 (Detailed Breakdown)

### 1. 早期尝试与基本单位的挑战
- 语音的取样点 (Sample) 和影像的像素 (Pixel) 太过细微。如 WaveNet 生成一秒声音需 90 分钟。
- 寻找适合的「基本单位」就像生物学找细胞一样。对于语音和影像，我们希望将相似的特征归类为同一个 Token。

### 2. 离散标记的局限性
- 透过编码器将影像/声音转为隐含向量 (Latent Representation)，再透过 Quantization 寻找最接近的字典 Token。
- 问题在于真实世界的声音与影像变化是连续的，强迫将它们归入有限的离散 Token 会导致失真与细节遗失。

### 3. 连续标记与生成模型的崛起
- 放弃离散化，直接使用连续向量 (Continuous Token)。
- 但传统接龙模型无法直接输出连续向量的机率分布。如果只用 MSE 训练，模型会倾向输出平均值（例如一只左眼看左边、一只左眼看右边的猫，平均起来变成三只眼睛的怪物），导致生成结果模糊。
- 引入 Flow Matching 或 Diffusion Model 来解决此问题。这些模型学习一个「向量场 (Vector Field)」或「去噪过程」，能够引导杂讯收敛到真实的资料分布，确保生成的特征是清晰且真实的。

### 4. 结合 Autoregressive 与 Generative Models
- 系统架构：首先用一个接龙模型，根据前面的上下文生成一个条件表示；接著，将这个条件输入给 Flow Matching/Diffusion 模型，将杂讯转换为精确的连续 Token；最后，透过解码器 (Detokenizer) 将这些 Token 还原成影像或声音。
- 这种结合使得模型既具备语言模型的强大上下文理解能力，又拥有扩散模型在连续资料上生成高画质/高音质的优势。

## 结论与未来展望 (Conclusions)
影像与语音生成的技术已经跨越了单纯的接龙时代，走向了更高效、更逼真的连续标记生成阶段。未来，生成式 AI 的发展将更加依赖多模态特征的融合，Diffusion 与 Autoregressive 模型的结合（如 Latent Diffusion Models 的进阶应用）将成为推动高拟真影音生成的关键驱动力。这使得我们距离能理解并创造出任意物理世界视觉与听觉体验的「世界模型 (World Models)」又更近了一步。
