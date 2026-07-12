# Claude Code 定制化全景：7 种控制方式

> **原始来源**: [Steering Claude Code: when to use CLAUDE.md, skills, hooks, rules, subagents and more](https://claude.com/blog/steering-claude-code-skills-hooks-rules-subagents-and-more)
>
> **学习日期**: 2026-07-12

---

## 概述

Claude Code 提供了 7 种定制化方式来控制 AI 编程助手的行为。理解每种方式的适用场景，是从"能用"到"高效用"的关键。

---

## 一、7 种定制方式速览

| # | 方式 | 定位 | 适用场景 |
|:--|:-----|:-----|:---------|
| 1 | **CLAUDE.md** | 记忆文件 | 项目约定、代码风格、常用命令 |
| 2 | **Rules** | 全局/组织级规则 | 跨项目共享的通用规范（如安全策略） |
| 3 | **Skills** | 可复用的程序化指令包 | 复杂流程、特定技术栈操作手册 |
| 4 | **Subagents** | 委派独立任务的子代理 | 并行工作、隔离上下文 |
| 5 | **Hooks** | 生命周期事件钩子 | 自动化门禁、CI/CD 集成 |
| 6 | **Output Styles** | 控制输出格式 | 简洁/详细/自定义响应风格 |
| 7 | **System Prompt Appending** | 附加系统提示词 | API/SDK 集成时注入行为指令 |

---

## 二、详细解析

### 1. CLAUDE.md — 项目记忆

**是什么**: 放在项目根目录（或 `~/.claude/` 全局目录）的 Markdown 文件，Claude 每次启动时自动读取。

**适合写什么**:
- 代码风格约定（缩进、命名规范）
- 常用的构建/测试/部署命令
- 项目架构说明和关键路径
- 工作流偏好

**示例**:
```markdown
# CLAUDE.md

## 构建命令
- 测试: `npm test`
- 构建: `npm run build`
- Lint: `npm run lint`

## 代码风格
- 使用 TypeScript strict 模式
- 函数命名: camelCase
- 组件命名: PascalCase
- 每个函数不超过 50 行
```

**层级关系**: 支持多级 `CLAUDE.md`：
- `~/.claude/CLAUDE.md` → 全局
- `项目根/CLAUDE.md` → 项目级
- `项目根/src/CLAUDE.md` → 目录级（当 Claude 访问该目录时生效）

---

### 2. Rules — 全局强制规则

**是什么**: 比 `CLAUDE.md` 更刚性的规则配置，适合组织级别的约束。

**与 CLAUDE.md 的区别**:
- CLAUDE.md 是"建议"，Rules 更接近"必须遵守"
- Rules 支持通过 managed policy 由组织管理员统一下发
- 适合安全策略、合规要求等不可协商的规则

---

### 3. Skills — 可复用的指令包

**是什么**: 包含 `SKILL.md` 的文件夹，定义了一套完整的操作流程。比 CLAUDE.md 更结构化，支持：
- YAML Frontmatter（名称、描述、甚至 Hooks）
- 附带脚本、示例、资源文件
- 按需激活（不是每次都加载）

**目录结构**:
```
.claude/skills/deploy/
├── SKILL.md          # 主指令文件（必需）
├── scripts/          # 辅助脚本
├── examples/         # 示例代码
└── resources/        # 资源文件
```

**核心优势**：Skills 可以在 Frontmatter 中注册 Hooks，实现"激活技能时自动挂载安全护栏，技能结束后自动卸载"。

---

### 4. Subagents — 子代理委派

**是什么**: Claude 可以启动独立的子代理来并行处理任务，每个子代理有自己的上下文窗口。

**适用场景**:
- 需要并行处理多个独立任务
- 任务需要大量上下文但不影响主对话
- 研究型任务（搜索、代码审查）

---

### 5. Hooks — 生命周期钩子

**是什么**: 绑定在 Claude 行为生命周期特定事件上的自动化执行器。

> 详见 [02_hooks_lifecycle_deep_dive.md](./02_hooks_lifecycle_deep_dive.md)

---

### 6. Output Styles — 输出风格

**是什么**: 控制 Claude 回复的风格和详细程度。
- 简洁模式：精简回答
- 详细模式：完整解释
- 自定义模式：按需调整

---

### 7. System Prompt Appending — 系统提示词注入

**是什么**: 通过 API 或 SDK 使用 Claude Code 时，在系统提示词中附加自定义指令。

**适用场景**: 将 Claude Code 集成到自定义工具链中。

---

## 三、如何选择？

```
需要 Claude 记住项目信息？ → CLAUDE.md
需要组织级强制规范？ → Rules
需要一套完整的操作流程？ → Skills
需要并行处理独立任务？ → Subagents
需要自动化的质量门禁？ → Hooks
需要控制输出格式？ → Output Styles
需要 API/SDK 级别的控制？ → System Prompt Appending
```

---

## 四、关键洞察

1. **这 7 种方式是互补的，不是互斥的**。一个成熟的项目可能同时用 CLAUDE.md + Skills + Hooks。
2. **Skills 是最强大的组合体**：它可以包含指令、脚本，甚至注册自己的 Hooks。
3. **Hooks 是确定性最强的控制方式**：不依赖 LLM 的"理解"，而是通过代码逻辑硬性执行。
