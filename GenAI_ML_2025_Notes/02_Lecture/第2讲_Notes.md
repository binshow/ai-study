# 机器学习 2025：第 2 讲 Context Engineering (上下文工程) 课程笔记

## 1. 执行摘要 (Executive Summary)
本讲探讨了当今 AI Agent 时代的关键技术：「Context Engineering (上下文工程)」。语言模型 (LLM) 的本质是「文字接龙 (预测下一个 Token)」，可视为一个函数 $f(x)$。若要改变输出结果，除了透过训练改变模型参数 $f$ (Training/Learning)，另一种方式就是改变输入 $x$，这正是 Context Engineering 的核心。

课程比较了早期的 Prompt Engineering (提示工程) 与现在的 Context Engineering 的差异，指出随著模型能力提升，过去依赖「神奇咒语 (Magic Spells)」的技巧已逐渐失效，现在的重点转向**如何自动化管理与优化输入给语言模型的 Context (上下文)**，特别是在 AI Agent 需要进行长时间、复杂任务规划时，如何「避免塞爆 Context」成为了最核心的挑战。为此，课程提出了三大 Context Engineering 的基本操作：**选择 (Selection)**、**压缩 (Compression)** 与 **Multi-Agent (多代理协作)**。

---

## 2. 核心概念 (Key Concepts)

*   **Prompt Engineering vs. Context Engineering**：两者本质相同，但关注点不同。前者过去常依赖特定格式或「神奇咒语」(如「一步一步思考」、「给你小费」)，而后者专注于自动化管理庞大的输入资讯。
*   **In-Context Learning (上下文学习)**：在不改变模型参数的前提下，透过在 Context 中提供范例或背景知识 (如字典、教科书)，让模型能完成原本不会的任务 (例如翻译罕见的卡拉蒙语)。
*   **Lost in the Middle & Context Rot**：尽管现代模型的 Context Window (上下文视窗) 动辄百万 Token，但输入过长会导致模型「发疯」或「迷失自我」，模型通常只记得开头与结尾的资讯，且随著长度增加，复制与理解能力会大幅下降。
*   **三大 Context 管理套路**：
    1.  **选择 (Selection)**：只挑选有用的资讯放入 Context (如 RAG、挑选工具、挑选记忆)。
    2.  **压缩 (Compression)**：将冗长的历史互动纪录进行摘要浓缩。
    3.  **Multi-Agent (多代理)**：将复杂任务与庞大 Context 分配给不同的 Agent 处理，避免单一 Agent 资讯超载。

---

## 3. 详细解析 (Detailed Breakdown)

### 3.1 一个完整的 Context 应该包含什么？
为了让语言模型产生正确的输出，Context 中通常会包含以下多种资讯：
1.  **使用者指令 (User Prompt)**：明确的任务说明、限制条件 (如字数、语气) 以及情境前提。
2.  **范例 (Examples)**：透过 In-Context Learning，给予清晰的输入/输出范例 (如火星文转换规则)。
3.  **系统提示 (System Prompt)**：由开发者设定的模型基础人设与行为准则 (如 Claude 3 Opus 长达 2500 字的 System Prompt，包含它是谁、限制事项、拒绝回答的规则等)。
4.  **对话历史纪录 (Conversation History)**：作为模型的「短期记忆」。
5.  **长期记忆 (Long-term Memory)**：跨对话的记忆 (如 ChatGPT 的记忆功能)，模型会自动从互动中提取并储存使用者资讯。
6.  **外部检索资讯 (RAG - Retrieval-Augmented Generation)**：搭配搜寻引擎或资料库获取的最新资讯。*(注：即使有 RAG，模型仍可能因搜寻到错误资讯而产生幻觉，如著名的「披萨加胶水」事件)*。
7.  **工具使用说明 (Tool Use / Computer Use)**：告诉模型有哪些工具可用 (如查询温度、操控滑鼠键盘) 以及其 JSON 或文字格式。模型会输出呼叫工具的指令文字，需由外部程式执行后再将结果放回 Context。
8.  **思考过程 (Reasoning/Thinking)**：如 OpenAI o1 或 DeepMind R 系列模型在给出答案前，自己产生的「脑内小剧场 (深度思考过程)」。

### 3.2 为什么 AI Agent 时代特别需要 Context Engineering？
*   **Agentic Workflow 的挑战**：AI Agent 需要自行决定解题步骤，并在过程中根据环境的回馈 (Observation) 不断采取行动 (Action)。
*   **Context Window 限制**：虽然 Gemini 1.5 号称有 200 万 Token、Llama 4 有 1000 万 Token，但**「能输入百万 Token，不代表能读懂百万 Token」**。
*   **效能退化**：给予过多资讯会导致模型头晕目眩，正确率先升后降。此外，挤牙膏式的冗长互动也会损害模型的能力。
*   **核心目标**：「避免塞爆 Context」，想办法只放需要的东西进去，清理掉不需要的内容。

### 3.3 Context Engineering 的三大基本操作
#### 1. 选择 (Selection)
*   **RAG 的延伸**：不只搜寻文章，还可以利用小模型做 Reranking，甚至只挑选「句子」进入 Context (如 Provence 论文)。
*   **挑选工具**：当工具多达上千个时，利用 RAG 的概念，只检索出与当前任务相关的工具说明给模型看，避免模型发疯 (Less is more)。
*   **挑选记忆**：如「史丹佛小镇」实验，AI 的记忆极其琐碎。系统会根据**最近程度 (Recency)**、**重要性 (Importance)** 与**相关性 (Relevance)** 进行评分，只把最高分的记忆放入 Context。
*   **挑选反馈经验 (SpringBench)**：研究发现，给模型「过去答对的经验」比「过去答错的经验」更有帮助。放入错误经验有时反而会让模型产生「白熊效应」，再次犯错。

#### 2. 压缩 (Compression)
*   **递回式压缩 (Recursive Compression)**：当互动历史过长时，呼叫另一个模型将过去的纪录做「摘要/压缩」。例如每互动 100 次就压缩一次，让远古的记忆留下大概的轮廓。
*   **处理 Computer Use 的琐碎资讯**：在使用电脑工具时 (如订餐厅)，会产生大量无用的滑鼠移动与视窗关闭纪录。将其压缩成「A 餐厅订位成功，9/19 下午 6 点，10人」即可。
*   **防遗忘机制**：若担心摘要遗失关键细节，可以将完整 Log 存入硬碟，并在摘要中留下索引 (如：「详细回忆请读取 xxx.txt」)，必要时再透过 RAG 唤回。

#### 3. Multi-Agent (多代理协作)
*   **Context 管理的利器**：Multi-Agent 不仅是因为每个 Agent 有不同的专长 (如 CEO、Programmer)，更是为了**分散 Context 负担**。
*   **资讯隔离**：
    *   总召 Agent 负责大方向规划。
    *   订餐厅 Agent 负责与网页互动，完成后只回报「餐厅已订好」给总召。
    *   总召的 Context 里只有高阶的任务状态，没有琐碎的操作细节，保持了 Context 的简洁与高效。
*   **平行处理大量资料**：如撰写 Overview Paper 时，让不同的 Agent 各自阅读少量的论文并写出摘要，最后再由一个 Agent 统整，避免单一 Agent 读取上千篇论文导致崩溃。研究显示，在复杂任务上 Multi-Agent 的表现远优于 Single-Agent。

---

## 4. 总结 (Conclusions)
随著语言模型基础能力的大幅提升，AI 应用的重点已从寻找字斟句酌的「神奇 Prompt」转变为系统化的「Context Engineering」。在让语言模型化身为能够长期自主运作的 AI Agent 时，如何有效管理其输入的 Context 是决定成败的关键。我们必须正视模型「Lost in the middle」与「Context Rot」的先天缺陷，善用**选择 (过滤无用资讯)**、**压缩 (摘要琐碎历史)** 与 **Multi-Agent (分工与资讯隔离)** 等架构设计，才能打造出稳定、聪明且能完成复杂现实任务的 AI Agent 系统。本课程不仅是一堂提示词的教学，更是 AI 系统架构设计的重要思维训练。