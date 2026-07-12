# Claude Code 定制化学习笔记 — 索引

> **学习日期**: 2026-07-12

---

## 原始来源

| # | 文章标题 | 链接 |
|:--|:---------|:-----|
| 1 | Steering Claude Code: when to use CLAUDE.md, skills, hooks, rules, subagents and more | [原文链接](https://claude.com/blog/steering-claude-code-skills-hooks-rules-subagents-and-more) |
| 2 | How to configure hooks | [原文链接](https://claude.com/blog/how-to-configure-hooks) |

---

## 笔记目录

| 文件 | 主题 | 核心内容 |
|:-----|:-----|:---------|
| [01_steering_claude_code_overview.md](./01_steering_claude_code_overview.md) | 定制化全景 | 7 种控制 Claude Code 的方式及选择策略 |
| [02_hooks_lifecycle_deep_dive.md](./02_hooks_lifecycle_deep_dive.md) | Hooks 生命周期深度解析 | 8 个生命周期事件、5 种 Hook 类型、Stop Hook 触发机制 |
| [03_skills_hooks_integration.md](./03_skills_hooks_integration.md) | Skills + Hooks 协同 | 在 Skill 中注册 Hook，实现"自带护栏的技能" |

---

## 核心概念速查

### 7 种定制方式
`CLAUDE.md` → `Rules` → `Skills` → `Subagents` → `Hooks` → `Output Styles` → `System Prompt Appending`

### 8 个 Hook 事件
`SessionStart` → `UserPromptSubmit` → `PreToolUse` → `PermissionRequest` → `PostToolUse` → `PreCompact` → `Stop` → `SubagentStop`

### 5 种 Hook 类型
`command` | `http` | `mcp_tool` | `prompt` | `agent`

### Hook 注册位置（作用域从大到小）
`Managed Policy`（组织强制） > `settings.json`（全局/项目） > `Skill/Agent Frontmatter`（按需激活）

---

## 关键 Q&A

**Q: Stop Hook 什么时候触发？**
A: 每次 Claude 完成一轮回答、准备把控制权交还用户时触发。一个 Session 中对话 N 轮就触发 N 次，不是 Session 结束才触发。

**Q: `type: "prompt"` 是什么？**
A: Hook 触发时不跑脚本，而是发起一次独立的轻量级 LLM 调用来做判断。适合无法用退出码（0/1）判定的模糊场景。

**Q: Skill 中可以使用 Hook 吗？**
A: 可以。在 `SKILL.md` 的 YAML Frontmatter 中声明 Hook，这些 Hook 只在该 Skill 被激活时生效，Skill 结束后自动卸载。

**Q: `prompt` 和 `agent` 类型 Hook 有什么区别？**
A: `prompt` 是一次无工具的轻量 LLM 调用，只能看和说；`agent` 启动一个有独立上下文和工具权限的完整 Agent，可以读文件跑命令。
