# Mini SWE Agent

CLI 工具，实现闭环 SWE agent：模型 → 解析恰好一条 Shell 动作 → 执行 → 观测写回 → 重复。

面向开发者与研究者：可测试、可记录轨迹、可审查、可选 Textual 检查器。

## 安全警告

**此工具会执行 AI 模型生成的任意 Shell 命令。** 这些命令可能修改或删除文件、安装软件、访问网络。在 Local 模式下不是玩具沙箱——请谨慎使用。如有可能，在 Docker 容器或沙箱中运行。

## 安装

```bash
pip install -e ".[dev]"
```

## 快速开始

```bash
# 运行单个任务（yolo 模式自动执行）
mswea run --task "修复 src/utils.py 中的除零错误" --model claude-sonnet-4-6

# confirm 模式（每步确认）
mswea run --task "添加单元测试" --confirm --step-limit 20

# 打印合并后的配置
mswea config -c model.name=gpt-4o

# 使用 Textual 检查轨迹
mswea inspect trajectory.traj.json

# 批量运行
mswea batch -f tasks.json -o preds.json -p 4
```

## CLI 选项

### `mswea run`

| 选项 | 描述 |
|------|------|
| `--task, -t TEXT` | 任务描述（必填） |
| `--model, -m TEXT` | 模型标识符 |
| `--config, -c PATH\|KEY=VALUE` | 配置文件或键值对（可重复） |
| `--yolo / --confirm` | 自动执行 / 每步确认（默认 yolo） |
| `--exit-immediately` | 达到终态后立即退出 |
| `--output, -o PATH` | 轨迹输出路径 |
| `--cost-limit FLOAT` | 最大费用（美元） |
| `--step-limit INT` | 最大步数 |
| `--help` | 显示帮助（含安全警告） |

### `mswea config`

打印合并后的完整配置并退出。接受 `--config` 参数。

### `mswea inspect PATH`

启动 Textual TUI 检查器查看 `.traj.json` 轨迹文件。

### `mswea batch`

| 选项 | 描述 |
|------|------|
| `--tasks-file, -f PATH` | JSON 任务文件（必填） |
| `--output, -o PATH` | 输出 preds.json 路径 |
| `--model, -m TEXT` | 模型标识符 |
| `--config, -c PATH\|KEY=VALUE` | 配置文件或键值对（可重复） |
| `--yolo / --confirm` | 自动执行 / 每步确认（默认 yolo） |
| `--cost-limit FLOAT` | 单任务最大费用（美元） |
| `--step-limit INT` | 单任务最大步数 |
| `--parallel, -p INT` | 并行任务数（默认 1） |
| `--regex-filter TEXT` | 对 instance_id 或 task 的正则过滤 |
| `--shuffle-seed INT` | 确定性 shuffle 种子 |
| `--slice START:END` | 任务切片（如 `0:10`） |
| `--redo-existing` | 重新运行 preds.json 中已有结果的任务 |

## 配置

支持多源配置，优先级从高到低：

1. CLI `--config k=v` 键值对
2. CLI `--config <文件>`（后指定覆盖先指定）
3. `./mswea.yaml`（工作目录）
4. `~/.config/mswea/config.yaml`
5. 内置 `defaults.yaml`

### 配置文件示例

```yaml
model:
  provider: anthropic
  name: claude-sonnet-4-6
  # API key 通过环境变量传入，不写在配置文件中
  # Anthropic SDK 会自动读取 ANTHROPIC_API_KEY 环境变量

limits:
  max_steps: 50
  max_cost: 10.0
  max_consecutive_format_errors: 3

executor:
  backend: local  # "local" 或 "docker"
  timeout: 120.0
  docker_image: "my-sandbox"  # docker 后端使用的容器名
```

API key 通过环境变量（`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`）传入，各 SDK 会自动读取。也可在配置文件中设置 `model.api_key`（不推荐，会泄漏至轨迹文件——轨迹 JSON 已自动排除该字段）。

## 动作协议

模型每步必须输出恰好 **一个** 可执行动作：

- **Tool-call 模式**：恰好一个 `bash` 工具调用，含 `command` 参数
- **文本模式**：恰好一个围栏块（`` ```mswea_bash_command `` 或 `<mswea_bash_command>`）

违反规则（0 个、>1 个、两族冲突）→ **FormatError**：不执行任何 Shell，向模型发送反馈消息。

## 提交契约

- 条件：`returncode == 0` **且** stdout 首行恰好为 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- `returncode != 0`：即使输出包含标记也**不予提交**

## 轨迹 JSON

格式含 `messages`（role/content/timestamp）和 `steps`（action/observation）。Tool-call 模式下观测与 `tool_call_id` 关联。

## 运行测试

```bash
# 单元测试 + 验收测试
pytest tests/ -v

# 真机/火测（需实际 API 调用）
MSWEA_LIVE_TESTS=1 pytest tests/acceptance/ -k live -v
```

## 环境变量

| 变量 | 描述 |
|------|------|
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 |
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `MSWEA_LIVE_TESTS=1` | 启用真机 API 测试 |

## 许可

MIT
