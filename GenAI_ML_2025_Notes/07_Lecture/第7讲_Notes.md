# 大型语言模型 (LLM) 学习历程：课程学习笔记

## 执行摘要 (Executive Summary)
本堂课程详细介绍了现代大型语言模型（如 ChatGPT、Gemini 等）的标准三阶段训练历程：预训练 (Pre-training)、监督式微调 (Supervised Fine-Tuning, SFT) 以及基于人类回馈的强化学习 (Reinforcement Learning with Human Feedback, RLHF)。这三个阶段本质上都在解决同一个问题——「文字接龙」（分类问题），但透过不同形式的训练资料与优化目标，使模型从一个单纯模仿网路文本的「学龄前儿童」，成长为懂得应对进退的「学生」，最终成为符合人类价值观且能举一反三的「社会化」人工智慧。

## 核心概念 (Key Concepts)
1. **文字接龙作为分类问题**：语言模型的核心机制是给定一段前文，预测下一个 Token 的机率分布。
2. **三阶段训练比喻**：
   - **Pre-training (学龄前)**：无忧无虑地吸收一切网路知识。
   - **SFT (上学)**：老师教导标准答案，学习人类对话的风格与规范。
   - **RLHF (出社会)**：接受社会的毒打与回馈（赞/倒赞），自行发掘对错。
3. **对齐 (Alignment)**：SFT 与 RLHF 的核心目的，皆是让机器的行为与输出对齐人类的需求与价值观。
4. **资料品质至上 (Quality is all you need)**：在 SFT 阶段，少量的极高品质资料（如 LIMA 仅用 1000 笔精挑细选的资料）甚至能胜过数百万笔低品质资料。
5. **知识蒸馏 (Knowledge Distillation)**：利用现有更强大的模型（如 GPT-4、Gemini）来生成 SFT 的标准答案，以降低人工标注的成本。
6. **RLHF 的优势**：相较于 SFT 逐字 (Token-level) 计算 Loss，RLHF 著眼于整个回答的完整性，并能够针对模型自己生成的答案「因材施教」。

## 详细内容解析 (Detailed Breakdown)

### 第一阶段：Pre-training（预训练）
* **任务本质**：Self-Supervised Learning（自督导学习），利用网路上无穷无尽的文本资料让机器学习文字接龙。
* **资料需求与处理**：
  * 需要天文数字般的资料量（例如 Llama 3 采用 15T tokens，印成 A4 纸叠起来高达 1500 公里，超越大气层）。
  * 网路资料充满杂讯，必须经过繁琐的清洗过程（Heuristic rules, Deduplication, Model-based filtering），最终可能只留下极低比例（如 1.4%）的高品质资料。
  * **目的**：让机器学习「语言知识」（语法）与「世界知识」（常识）。同一个实体（Entity）需在资料中以多种不同角度反复出现，模型才能真正将其融会贯通并举一反三。
* **阶段结果**：产生 Base Model。此时的模型拥有庞大知识，但不知如何正确回答问题（直接对其提问可能会回问问题或输出选择题）。但 Base Model 潜力无穷，是后续 SFT 与 RLHF 的 Initialization（初始参数）。

### 第二阶段：Supervised Fine-Tuning, SFT（监督式微调）
* **任务本质**：Supervised Learning，又称 Instruction Fine-Tuning。人类提供明确的「指令 (Instruction)」与「标准答案」，教导机器如何使用 Chat Template 并给予正确回应。
* **关键发现**：
  * **SFT 不产生新知识，而是改变输出风格**：模型的知识在预训练阶段已基本固定。SFT 只是「画龙点睛」，引导模型去检索并正确输出早已学会的知识。
  * **资料重质不重量**：使用数万笔甚至精挑细选的 1,000 笔高品质资料，往往比使用几百万笔粗糙资料的表现更好。
* **变体与进阶技巧**：
  * **Knowledge Distillation**：拿强大的现成模型（Teacher，如 ChatGPT、Gemini）产生答案来教导自己的模型（Student），如 Alpaca 和 Vicuna 的做法。
  * **Response Tuning**：甚至不需要提供完整的指令与问题，仅提供高质量的回答让模型学习，也能激发模型 Instruction Following 的能力。

### 第三阶段：Reinforcement Learning with Human Feedback, RLHF（人类回馈强化学习）
* **任务本质**：在没有绝对标准答案的情况下，根据人类对模型生成完整回答的满意度（Reward：通常为赞或倒赞）来更新模型。
* **为什么需要 RLHF 而非只有 SFT？**
  1. **全局视角 vs 逐字比较**：SFT 计算每个 Token 的 Loss，可能导致句意不通但整体 Loss 较低的状况；RL 则是对「整个回答」进行评分，更贴近人类的真实偏好。
  2. **因材施教**：SFT 的标准答案是由人类写的，对机器可能太难或思路不同（老师教的不一定是学生想学的）；RL 则是让机器先自己产出答案，人类再给予回馈，机器能针对自己的痛点进行学习。
* **演算法精神 (如 Policy Gradient)**：
  * 得到正面回馈 (Positive Feedback) 时，拉近模型输出与该答案的距离（Maximize 机率，与一般 Supervised Learning 相似）。
  * 得到负面回馈 (Negative Feedback) 时，增加一个负号，拉远模型输出与该错误答案的距离。
* **RLAIF (AI 回馈强化学习)**：
  * 因人类标注成本太高，目前常见作法是训练一个「Reward Model（评分模型）」来代替人类给分。
  * 甚至可以让语言模型自己评分（Self-rewarding），实现无穷无尽的自动回馈学习。

## 结论 (Conclusions)
大型语言模型的成功并非单一技术的突破，而是 Pre-training、SFT、RLHF 三阶段紧密相连的结果。
1. **Pre-training** 提供了模型不可或缺的庞大语言与世界知识底蕴（站在巨人的肩膀上）。
2. **SFT** 透过极高的资料品质，有效引导模型学会了人类的沟通模式，发挥其既有的知识潜力（画龙点睛）。
3. **RLHF / RLAIF** 则突破了有标准答案学习的局限，透过全局回馈让模型自我修正与学习，最终达成对齐（Alignment）人类价值观的终极目标。

这三个阶段虽然训练的资料来源与 Loss 函数的定义有所不同，但本质上皆是以「文字接龙」为核心，透过 Gradient Descent 等优化方法，造就了今日强大且实用的人工智慧模型。