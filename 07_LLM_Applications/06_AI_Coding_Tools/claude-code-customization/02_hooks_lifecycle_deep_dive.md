# Claude Code Hooks 生命周期深度解析

> **原始来源**: [How to configure hooks](https://claude.com/blog/how-to-configure-hooks)
>
> **学习日期**: 2026-07-12

---

## 概述

Hooks 是 Claude Code 中确定性最强的控制机制。它们不依赖 LLM "理解你的意思"，而是在明确的生命周期事件上，确定性地执行你定义的操作。

---

## 一、8 个生命周期事件

```
┌───────────────────────────────────────────────────────────┐
│                   Claude Code Session                      │
│                                                           │
│  🟢 SessionStart ─── 会话初始化时触发                       │
│                                                           │
│  ┌─────────── 用户输入一条消息 ───────────┐                │
│  │                                       │                │
│  │  📝 UserPromptSubmit                  │                │
│  │     用户提交 prompt 后、Claude 处理前   │                │
│  │                                       │                │
│  │  ┌──── Claude 开始工作循环 ────┐       │                │
│  │  │                           │       │                │
│  │  │  🔒 PermissionRequest     │       │                │
│  │  │     需要用户授权时         │       │                │
│  │  │                           │       │                │
│  │  │  ⚡ PreToolUse             │       │                │
│  │  │     调用工具之前           │       │                │
│  │  │                           │       │                │
│  │  │  📋 PostToolUse            │       │                │
│  │  │     工具调用完成后         │       │                │
│  │  │                           │       │                │
│  │  │  📦 PreCompact             │       │                │
│  │  │     上下文压缩之前         │       │                │
│  │  │                           │       │                │
│  │  └───────────────────────────┘       │                │
│  │                                       │                │
│  │  🛑 Stop                              │                │
│  │     Claude 完成本轮回答时              │                │
│  │                                       │                │
│  └───────────────────────────────────────┘                │
│                                                           │
│  🤖 SubagentStop ─── 子代理完成任务时触发                   │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

## 二、各事件详解

### 1. SessionStart
- **触发时机**: Claude Code 启动或新会话开始时
- **典型用法**: 环境检查、依赖验证、工作区初始化
- **示例**: 检查 Node.js 版本、验证 API Key 是否配置

### 2. UserPromptSubmit
- **触发时机**: 用户提交消息后、Claude 开始处理前
- **典型用法**: 输入预处理、日志记录、合规审计
- **示例**: 记录所有用户指令到日志文件

### 3. PreToolUse
- **触发时机**: Claude 准备调用工具之前
- **典型用法**: 拦截危险操作、参数校验、权限控制
- **示例**: 禁止写入 `*.prod.env` 文件、阻止删除操作
- **Hook 可返回**: `approve`（放行）、`deny`（拒绝）、修改后的参数

### 4. PermissionRequest
- **触发时机**: Claude 遇到需要用户授权的操作时
- **典型用法**: 自动审批策略、记录审批日志
- **示例**: 自动批准所有读操作，但拦截写操作

### 5. PostToolUse
- **触发时机**: 工具调用完成后
- **典型用法**: 结果验证、自动格式化、增量测试
- **示例**: 每次写文件后自动运行 `prettier` 格式化

### 6. PreCompact
- **触发时机**: 上下文窗口即将压缩之前
- **典型用法**: 保存关键信息、导出上下文快照
- **示例**: 将当前工作进度保存到文件

### 7. Stop ⭐
- **触发时机**: **Claude 完成本轮回答、准备交还控制权时**
- **⚠️ 关键**: 一个 Session 中用户对话 N 轮，Stop 就触发 N 次
- **典型用法**: 自动验证、自主 Agent Loop
- **特殊能力**: Hook 返回 `"continue": true` 可让 Claude 继续工作而不交还控制权

### 8. SubagentStop
- **触发时机**: 子代理完成任务时
- **典型用法**: 子代理结果验证、结果汇总
- **示例**: 验证子代理的代码修改是否通过测试

---

## 三、Hook 的 5 种执行类型（type）

| type | 执行方式 | 确定性 | 工具权限 | 适用场景 |
|:-----|:---------|:-------|:---------|:---------|
| `command` | 运行本地 Shell 命令 | ✅ 确定性 | N/A | lint、test、git 操作 |
| `http` | 发送 HTTP 请求 | ✅ 确定性 | N/A | Webhook 通知、API 调用 |
| `mcp_tool` | 调用 MCP 工具 | ✅ 确定性 | N/A | 集成外部工具服务 |
| **`prompt`** | **发起独立 LLM 调用做判断** | ⚡ 非确定性 | ❌ 无 | 模糊场景的快速是/否裁决 |
| **`agent`** | **启动独立 Agent** | ⚡ 非确定性 | ✅ 有 | 需要读文件/跑命令的复杂判断 |

### `prompt` vs `agent` 对比

```
prompt 类型:
  触发 → 发起一次轻量 LLM 调用 → 返回判断结果
  特点: 快速、无工具权限、只能"看"和"说"

agent 类型:
  触发 → 启动独立 Agent（有自己的上下文窗口和工具） → 执行操作 → 返回结果
  特点: 强大、有工具权限、可以读文件跑命令
```

---

## 四、配置方式

Hooks 可以在以下 3 个位置注册：

### 1. settings.json（全局生效）

文件位置: `~/.claude/settings.json`（用户级）或 `.claude/settings.json`（项目级）

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "npx prettier --write $CLAUDE_FILE_PATH"
          }
        ]
      }
    ]
  }
}
```

### 2. Skill Frontmatter（技能激活时生效）

```markdown
---
name: deploy
description: 部署流程
hooks:
  Stop:
    - hooks:
        - type: prompt
          prompt: "检查部署是否完成"
---
```

### 3. Managed Policy（组织管理员下发，不可覆盖）

由管理员统一配置，确保安全策略在所有用户环境中强制执行。

---

## 五、Stop Hook 深度解析

### 触发时机

```
用户提问 1
  → Claude 工作 → 回答完毕
  → 🔔 Stop Hook 触发（第 1 次）
  → 控制权交还用户

用户提问 2
  → Claude 工作 → 回答完毕
  → 🔔 Stop Hook 触发（第 2 次）
  → 控制权交还用户

... 每轮都触发，不是 Session 结束才触发
```

### 自主 Agent Loop 模式

当 Stop Hook 返回 `"continue": true` 时，Claude 不会停下来等用户输入，而是继续工作：

```
用户: "重构整个认证模块"
  → Claude 第 1 轮工作完成
  → Stop Hook 触发 → prompt 类型 LLM 判断: "continue — 还有测试没写"
  → Claude 继续第 2 轮工作
  → Stop Hook 触发 → prompt 类型 LLM 判断: "continue — 文档未更新"
  → Claude 继续第 3 轮工作
  → Stop Hook 触发 → prompt 类型 LLM 判断: "complete"
  → 真正停下，交还控制权
```

### ⚠️ 注意事项

因为 Stop 每轮都触发，需要注意：
1. **避免重操作**: 不要无条件跑完整测试套件，否则每轮回答都会等测试跑完
2. **加判断逻辑**: 用 `git diff --name-only` 检查是否有实际改动，没改动就跳过
3. **优先用 `prompt` 类型**: 让 LLM 轻量判断是否需要执行重操作，而不是无脑执行

---

## 六、实际应用案例

### 案例 1: 自动格式化（PostToolUse + command）
每次写文件后自动运行 Prettier。

### 案例 2: 安全护栏（PreToolUse + command）
禁止修改生产环境配置文件。

### 案例 3: 自主任务完成验证（Stop + prompt）
让 LLM 判断任务是否真正完成，未完成则让 Claude 继续。

### 案例 4: 技能级 Hook（Skill Frontmatter）
在 Skill 中注册 Hook，技能激活时自动挂载，结束后自动卸载。
