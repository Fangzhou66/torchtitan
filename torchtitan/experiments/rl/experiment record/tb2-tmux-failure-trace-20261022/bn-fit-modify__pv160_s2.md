# 一个"装 tmux 失败"的 eval rollout（人类可读版）

## 基本信息

| 项 | 值 |
|---|---|
| Run | 1021（`q35-9b-agentpick-cleancf-evolve-both-1021-5aa7fe`，attempt 1） |
| Eval 点 | pv160（训练里那次 TB2.1 eval，89 题 × 5 次） |
| 题目 | `bn-fit-modify`，第 3 次尝试（trial `bn-fit-modify__g-536_s2`，日志里是 `group=-180/rollout=2`） |
| 镜像 | `docker.io/alexgshaw/bn-fit-modify:20251031`（TB2 官方镜像，**里面没有 tmux**） |
| Sandbox | Daytona `da604521-7bf1-439a-9c3a-344e3f57eba3`，disk 10 GB |
| 结果 | `INFRA_FAILED`，模型 **0 轮**，eval 记 **0 分** |
| 耗时 | 共 218 s，其中执行命令 207 s（11 条命令） |

## 发生了什么（按时间顺序）

1. **Sandbox 创建成功，网络正常**

   开机自检（`tmax_net_probe`）用了 0.6 s，外网全通：

   ```
   github.com       DNS ok 7ms   HTTP 200 114ms
   huggingface.co   DNS ok 4ms   HTTP 200  55ms
   pypi.org         DNS ok 4ms   HTTP 200  57ms
   ```

2. **Harness 开始给模型准备终端**（`[terminus] session=group=-180/rollout=2 stage=setup start`）

   Terminus-2 需要 tmux 当终端。harbor 的 `tmux_session.py` 依次执行：

   | 步 | 命令 | 结果 |
   |---|---|---|
   | a | `tmux -V` | exit 127：`tmux: command not found`（镜像里没有 tmux） |
   | b | 探测系统和包管理器 | 找到 `apt-get` |
   | c | `DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y tmux` | **超时**：`exit_code 124`，"Command timed out after 120s." |

   第 c 步对应的 Daytona 事件，是这个 rollout 唯一的一条 sandbox 事件：

   ```json
   {"event": "sandbox_issue", "kind": "command_timeout", "exit_code": 124,
    "message": "Command timed out after 120s.", "phase": "command",
    "instance_id": "bn-fit-modify", "group_id": -180, "rollout_id": 2,
    "sandbox_id": "da604521-7bf1-439a-9c3a-344e3f57eba3", "recovered": false}
   ```

   harbor 只在"apt-get **成功**但 tmux 还是找不到"时才会改为从源码编译。这里 apt-get 是**超时失败**，所以不会尝试从源码编译，直接进入下一步。

3. **启动 tmux 会话失败**

   ```
   tmux new-session -x 160 -y ... -d -s ...   ->  exit 127  (tmux: command not found)
   ```

   harbor 抛出：

   ```
   RuntimeError: Failed to start tmux session. Error:
   ```

   `Error:` 后面是空的，因为 "command not found" 打在 stdout，而 harbor 只拼接了 stderr。

   调用栈：

   ```
   rollouter.py:2060  _run_agent_rollout
   terminus.py:1041   terminus_agent -> agent.setup(env)
   terminus_2.py:372  setup -> self._session.start()
   tmux_session.py:435 start -> raise RuntimeError("Failed to start tmux session. Error: ...")
   ```

4. **Rollout 收尾**

   ```
   [tmax] group=-180/rollout=2: rollout failed
   [tmax] group=-180/rollout=2: status=error reward=0.00 turns=0 oom_suspect=0
          secs=218 exec_secs=207 exec_n=11 budget=4800
   ```

   最终汇总（`tmax_sandbox_summary`）：

   ```json
   {"infra_failed": true, "finish_reason": "error", "submitted": false, "reward": 0.0,
    "failure": {"origin": "unknown", "reason": "RuntimeError", "stage": "agent"},
    "issue_counts": {"command_timeout": 1}}
   ```

   eval trace 文件（`validation_traces/step-160/traces/bn-fit-modify/bn-fit-modify__g-536_s2.md`）里也只记了这些：

   ```
   | State          | INFRA_FAILED |
   | Rollout status | error |
   | failure        | {'origin': 'unknown', 'reason': 'RuntimeError', 'stage': 'agent'} |
   | finish_reason  | error |
   | infra_failed   | True |
   ```

## 为什么会超时

- 这次 eval 的 445 个 sandbox 几乎同时启动（eval 启动并发 445），每个都要执行 `apt-get update && apt-get install tmux`，同时去 Ubuntu 镜像源下载。
- 这个命令的 p90 耗时在 1021 里一路变长：pv40 是 11 s，pv140 涨到 64 s，最慢的那批超过 120 s 就失败。网络本身是通的（见第 1 步），慢在镜像源或出口被挤满。
- 1021 的 eval 还和训练共用 Daytona（训练约 2000 个并发 rollout）。
- 1021 中这类失败每次 eval 的个数：pv80 5、pv100 11、pv120 17、**pv140 35**、pv160 11。1018 是 0 次。

## 对分数的影响

- 在 eval 里，infra 失败按 0 分算，分母不变。这次尝试模型其实一轮都没开始，所以这个 0 分和模型能力无关。
- 把 eval 启动并发降到 225（单独跑 eval 的那三次），tmux 失败降为 0。

## 修复办法（只改 eval）

1. 把 tmux 预装进 eval 镜像或 snapshot，彻底去掉启动时的 apt-get。
2. 降低 eval 启动并发，例如 225 分两波，或者把 sandbox 创建分散到 3–5 分钟内。
3. 给 tmux 安装加更长的超时并带退避重试；另外让报错信息带上 stdout，不然 `Error:` 后面永远是空的。
