# Skills 与 Hooks 的协同：自带护栏的技能

> **原始来源**:
> - [Steering Claude Code](https://claude.com/blog/steering-claude-code-skills-hooks-rules-subagents-and-more)
> - [How to configure hooks](https://claude.com/blog/how-to-configure-hooks)
>
> **学习日期**: 2026-07-12

---

## 概述

Claude Code 的 Skills 不仅是一套文字指令，还可以在 YAML Frontmatter 中注册 Hooks。这意味着 **Skill 可以自带安全护栏和自动化逻辑**，只在该 Skill 被激活时生效。

---

## 一、核心机制

文章原文明确指出：

> "You register hooks in `settings.json`, managed policy settings, or **skill/agent frontmatter**."

Hooks 的三种注册位置对应三种作用域：

| 注册位置 | 作用范围 | 生命周期 |
|:---------|:---------|:---------|
| `settings.json` | 全局，所有会话中每轮都生效 | 永久 |
| **Skill Frontmatter** | **仅当该 Skill 被激活时生效** | 随 Skill 激活/结束 |
| Agent Frontmatter | 仅当该 Subagent 被唤醒时生效 | 随 Subagent 激活/结束 |

---

## 二、Skill + Hook 实战示例

### 示例 1: 部署技能（带质量门禁）

```
.claude/skills/deploy/
├── SKILL.md
├── scripts/
│   └── pre_deploy_check.sh
└── examples/
    └── deploy_config.yaml
```

**SKILL.md**:
```markdown
---
name: deploy
description: 执行生产环境部署流程
hooks:
  PreToolUse:
    - matcher: "Write|Edit"
      hooks:
        - type: command
          command: "echo $CLAUDE_FILE_PATH | grep -q '.env.production' && echo 'deny: 禁止修改生产环境变量' || echo 'approve'"
  PostToolUse:
    - matcher: "Write"
      hooks:
        - type: command
          command: "npm run lint -- --quiet"
  Stop:
    - hooks:
        - type: prompt
          prompt: "检查部署清单：1) 所有测试通过 2) 环境变量配置正确 3) 构建成功。若全部满足回复 complete，否则回复 continue 并说明遗漏项。"
---

## 部署流程

1. 验证所有环境变量已正确配置
2. 运行完整测试套件
3. 执行生产构建
4. 生成部署清单
5. 推送到生产环境
```

**效果**：
- ✅ 激活 `deploy` 技能时：自动阻止修改 `.env.production`，每次写文件自动 lint，完成时自动验证部署清单
- ✅ 普通对话时：上述限制完全不存在，零开销

### 示例 2: 代码审查技能（带自动测试）

```markdown
---
name: code-review
description: 深度代码审查
hooks:
  Stop:
    - hooks:
        - type: command
          command: "git diff --name-only | xargs -I {} sh -c 'echo \"Changed: {}\"' && npm test 2>&1 | tail -5"
---

## 审查流程

1. 阅读 PR 中所有变更文件
2. 检查代码风格和最佳实践
3. 验证边界条件和错误处理
4. 确认测试覆盖
```

---

## 三、设计原则

### 1. Hook 逻辑要轻量
Skill 中的 Hook 每轮都可能触发，避免在 Hook 中执行耗时操作。优先使用 `prompt` 类型做快速判断，必要时才触发重操作。

### 2. 用 matcher 精确匹配
不要在所有工具调用上都挂 Hook，用 `matcher` 字段精确匹配目标工具（如只在 `Write` 或 `Edit` 时触发）。

### 3. 安全护栏跟着技能走
将安全限制封装在 Skill 中，而不是放在全局 `settings.json`。这样：
- 不会影响日常开发体验
- 只在高危操作（如部署）时自动启用
- 技能结束后自动解除

### 4. 组合使用多种 Hook 类型
一个 Skill 可以同时注册多个事件的 Hook：
- `PreToolUse` → 拦截危险操作
- `PostToolUse` → 自动验证
- `Stop` → 确认任务完成

---

## 四、关键洞察

1. **Skill 是最强大的组合体**: 指令 + 脚本 + Hooks，三者合一。
2. **作用域隔离是杀手级特性**: Hook 只在 Skill 激活时生效，避免了全局 Hook 的性能开销和干扰。
3. **"自带护栏的技能"范式**: 把安全策略和自动化逻辑直接绑定到操作流程中，而不是依赖外部的全局配置。
4. **`prompt` 类型 Hook 是粘合剂**: 在 Skill 的 Stop Hook 中使用 `prompt` 类型，可以让 AI 智能判断复杂任务是否真正完成。
