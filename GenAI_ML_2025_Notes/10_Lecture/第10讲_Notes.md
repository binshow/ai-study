# 语音语言模型发展史 (History of Speech Language Models)

## 1. 执行摘要 (Executive Summary)
本讲为生成式 AI 与机器学习课程的最后一堂课，由李宏毅老师以第一人称视角，带领探讨语音语言模型（Speech Language Model, SLM）的发展历程。课程从早期的语音辨识技术（如 Cascade 模型与 End-to-End 模型的竞争）出发，探讨语音表征模型（Speech Representation Models）的崛起，并介绍初代语音语言模型的架构（Tokenizer, LM, Detokenizer）。最后，课程深入探讨了最新的语音语言模型技术，包括如何让模型「边说边想」（STITCH 架构），以及未来全双工（Full Duplex）对话与物理时间感知等未解难题。

## 2. 核心概念 (Key Concepts)
- **语音语言模型 (Speech Language Model)**：输入语音并输出语音的模型（如 ChatGPT Voice Mode, Gemini Live）。主要分为对话模式（Dialog Mode）与指令模式（Command Mode）。
- **Cascade vs. End-to-End ASR**：早期语音辨识多采串联（Cascade）多个模组的方式；端到端（End-to-End）模型虽然初期表现不佳，但经过近10年发展，已能超越 Cascade 模型。
- **无监督语音辨识 (Unsupervised ASR)**：利用自监督学习（Self-Supervised Learning）产生的语音表征向量（Audio Word Vector），结合 GAN 等对抗生成技术，在不依赖成对标注资料的情况下，将语音转换为文字。
- **初代语音语言模型架构 (如 GSLM)**：
  - **Tokenizer**：将语音压缩、离散化成 Token（如 Neural Speech Codec）。
  - **Language Model**：自回归模型，负责预测下一个 Token。
  - **Detokenizer**：将生成的 Token 还原（解压缩）为语音讯号。
- **STITCH 模型**：利用语音生成运算与语音播放之间的时间差（Buffer time），让模型在生成语音的空档中生成「文字推理（Reasoning）」Token，达成「边说话边思考」的零延迟推理体验。
- **全双工对话 (Full Duplex)**：超越回合制的对话模式，允许双方同时发声、插话或给予反馈，是语音模型未来追求的重要发展方向。

## 3. 课程详细内容 (Detailed Breakdown)

### 3.1 语音语言模型的定义与模式
- **定义**：有别于常见的文字输入输出模型，语音语言模型直接处理声音讯号（Speech in, Speech out），且并非单纯的语音辨识（ASR）或语音合成（TTS）。
- **两种主要模式**：
  - **对话模式 (Dialog Mode)**：将输入视为对话，给予相应的语音回复（如：听到 "How are you"，回答 "I'm fine"）。
  - **指令模式 (Command Mode / Speech Aware LM)**：给定一段语音及一个文字指令，模型根据指令处理该语音（如：指令为「翻译」，听到 "How are you"，回答「你好」）。
- 目前学界名词使用尚未统一，显示此领域仍在快速发展的探索阶段。

### 3.2 寒武纪之前：Deep Learning 与 End-to-End ASR 的演进
- **早期 Deep Learning 在语音的应用 (2010)**：Geoffrey Hinton 等人将 Deep Learning (RBM) 应用于音素辨识（Phone Recognition）。当时错误率约 26.7%，并未超越传统的 HMM 模型（24.8%），在 ICASSP 2010 上并未受到广泛关注。
- **End-to-End 模型的逆袭**：2015年前后，端到端系统（直接从语音转文字）刚出现时效果极差。但经过十年的努力与技术迭代，端到端模型最终超越了 Cascade 模型，主导了整个语音社群。这说明新技术的成熟需要长时间的孵化。

### 3.3 语音表征模型与无监督语音辨识 (Unsupervised ASR)
- **Segmental Audio Word Vector (2015)**：
  - 模型能自动决定语音的切分点（Segment），并将每个片段转换为一个向量。
  - 训练目标：利用 LSTM Decoder 将向量还原为原始语音，同时限制切分点的数量越少越好。
- **无监督语音辨识的突破**：
  - 将上述无监督学习到的语音向量，透过 GAN 的概念与无关的文本资料（Text Corpus）对齐，实现在无标注数据下的语音辨识。
  - 后续 Meta 提出的 **wav2vec 2.0** 取出特定隐藏层（如第15层）特征，让无监督语音辨识的错误率暴跌至 15% 左右（在 TIMIT 资料集上几乎追平有监督学习）。进阶版本如 **wav2vec-U 2.0** 更在 LibriSpeech 达到 5.4% 的极低错误率。

### 3.4 语音问答系统 (Speech QA) 的发展
- **早期尝试**：2016年将托福听力测验视为分类问题，正确率仅40多%，虽然优于随机盲猜，但远不及人类水平。
- **Dual 模型 (End-to-End Speech QA)**：在有了强大的 Speech Representation Model（如 HuBERT）后，可跳过语音辨识（ASR）步骤，直接将语音表征输入给 Downstream 模型生成答案。当 ASR 错误率高于 25% 时，End-to-End QA 模型的表现反而优于 Cascade（ASR + 文字 QA）模型。

### 3.5 石器时代：初代语音语言模型 (如 GSLM)
- 初代模型架构由三部分组成：
  1. **Tokenizer (Codec)**：将连续语音转为离散 Token。早期直接对 Representation 向量做 K-Means 分群；后来发展出 Neural Speech Codec，将 Tokenizer 与 Detokenizer 联合训练。
  2. **Autoregressive Model**：与文字语言模型无异，负责 Token 序列的接龙生成。
  3. **Detokenizer**：将 Token 解压缩（Decompress）回语音波形。
- **Textless NLP**：探讨纯语音 Token 是否能包含语义？研究表明，适当设计的语音 Token 可以控制语速、情绪等声学特征，并继承部分语言模型的语义能力。
- **李宏毅团队的实验模型**：
  - 将大量无标注语音利用 ASR 转为文字，交错输入文字与语音 Token 训练模型。
  - 模型初始化自文字语言模型，以继承其强大的文字能力。
  - 评估方式包含语义连贯性（交由 GPT-4 评分）及音质（UTMOS）。
  - 实验证明其在中文语音接龙上有著自然的发音，但因仅做 Pre-training，会随机产生幻觉（Hallucination）。

### 3.6 现代突破：STITCH 边说边想模型
- **问题痛点**：传统加入推理（Reasoning）的方法会让模型思考完才发声，导致语音对话中出现极不自然的漫长等待延迟。
- **发现与解法 (STITCH)**：
  - 语音语言模型生成 Token 的速度远快于音档播放的速度（例如：用 A100 生成 39 个语音 Token 只需 0.5 秒，但播放该段声音需 2 秒）。
  - 这产生了 1.5 秒的运算空档（Buffer）。STITCH 模型利用这段空档，让模型生成「文字 Reasoning Token」。
- **运作流程**：模型先讲出第一段语音（如重复使用者的问题争取时间），在播放这段语音的同时，内部快速生成推理 Token，再接著生成下一段回复的语音 Token。
- **实验成果**：在数学解题数据集上，STITCH 将正确率从 53% 提升至 78%，且达到了**零感知延迟**的自然互动体验。

### 3.7 全双工与未解难题
- **全双工对话 (Full Duplex)**：未来的语音模型需支援双向同时沟通，包含：
  - 适时给予反馈（如：在对方讲话时发出「嗯嗯」、「对」等声音）。
  - 处理使用者的打断与插话，或主动打断使用者。
- **物理时间感知**：目前的模型缺乏对物理世界时间流逝的概念（例如无法执行「安静 10 秒钟」的指令），这是让 AI 真正融入现实世界的重要研究方向。

## 4. 结论与未来展望 (Conclusions & Future Outlook)
语音语言模型经历了从 Cascade 模组化到 End-to-End 联合训练的典范转移。得益于自监督学习与神经语音编码器（Neural Speech Codec）的进步，如今的模型已能直接处理语音输入与输出。透过如 STITCH 这样的创新架构，模型更能在不牺牲互动延迟的情况下执行复杂推理。然而，如何实现自然流畅的全双工对话，以及赋予模型物理时间意识，仍是研究人员未来需努力克服的挑战。
