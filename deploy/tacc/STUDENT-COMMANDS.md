# CSIT6000V Project：全公共环境，官方 SDK 提交

解压本材料到自己的客户端工作目录。公共环境和模型已经安装，无须复制环境或下载权重。

在能够访问平台网关的客户端安装官方原版 SDK，使用自己的账号和已登记公钥对应的私钥：

```bash
tcloud config -u YOUR_USERNAME -f /path/to/your/private_key
tcloud init
tcloud env ls
```

平台应已提供个人 `~/.Miniconda3`。先运行 `bash setup-sdk-account.sh`，复用作业1已配置的课程 SDK 管理入口。该脚本仅通过原版 SDK 配置公共环境搜索路径和个人空索引，不安装软件包。配置中的 `environment.name: mpi-gemm-v1` 仅用于 SDK 的管理和激活步骤；`nodefaults` 与空依赖映射 `"{}"` 不提出新增软件包需求。任务入口随后激活课程环境，实际服务和程序使用课程 Python。学生电脑无需安装 CUDA、PyTorch、Conda 或 vLLM。

## 启动模型服务

```bash
cd qwen-public-serve
tcloud submit
```

下面的相对路径均从材料包根目录开始；进入任务目录提交后，返回根目录再编辑另一个任务。

记录提交返回的 JOB_ID，查看状态和日志：

```bash
tcloud ps -j JOB_ID
tcloud cat slurm_log/qwen-public-serve-JOB_ID.log
```

等待日志出现 `Application startup complete`。`SERVICE_BASE_URL=http://NODE:8000/v1` 给出服务地址。作业运行状态不代表服务就绪；首次加载、即时编译和预热需要数分钟，均发生在平台计算作业内。

在本地客户端编辑 `qwen-public-client/server-url.txt`，写入本次服务地址，再提交自己的程序：

```bash
printf '%s\n' 'http://NODE:8000/v1' > qwen-public-client/server-url.txt
cd qwen-public-client
tcloud submit
```

客户端另占 1 CPU，不申请 GPU。记录客户端作业 CLIENT_JOB_ID 后：

```bash
tcloud ps -j CLIENT_JOB_ID
tcloud cat slurm_log/qwen-public-client-CLIENT_JOB_ID.log
tcloud download slurm_log/qwen-public-client-CLIENT_JOB_ID.log ./client-result.log
```

例子应输出 `95` 和 `北京`。自己的程序放在 `qwen-public-client/main.py`，可以 `from qwen_client import chat` 调用服务；其他源文件放在同一客户端任务目录。服务与客户端是两个独立 SDK 作业，使用相同的课程依赖，各有独立运行时临时目录。客户端运行期间服务必须保持运行。

完成后用服务作业编号停止服务：

```bash
tcloud cancel -j JOB_ID
tcloud ps -j JOB_ID
```

模型名 `qwen3.8-27b`，端口 8000，32K 上下文、4 并发、仅文本；服务申请单节点 2 GPU、8 CPU、2 小时。作业结束后地址失效，下次启动重新设置节点地址。个人电脑默认通过 SDK 提交客户端程序到平台调用；如果需要从校外电脑直接访问 HTTP 服务，请使用平台明确提供或批准的访问入口。

## 文件与环境

- `qwen-public-serve/tuxiv.conf`：GPU 服务资源和入口。
- `qwen-public-serve/main.py`、`job.sh`：激活公共环境并启动现有服务脚本。
- `qwen-public-client/tuxiv.conf`、`job.sh`：CPU 客户端资源和同一公共环境入口。
- `qwen-public-client/main.py`、`qwen_client.py`：学生程序示例和 HTTP 调用函数。
- `qwen-public-client/server-url.txt`：本次运行服务的地址。

公共激活入口 `/mnt/home/public/course-envs/CSIT6000V/project/activate.sh`。公共依赖只读；学生代码和结果由 SDK 放在个人目录。SDK 会同步各任务的临时 WORKDIR，持久结果保存在 USERDIR 或 slurm_log。不要手动编辑 SDK 自动生成的 `run.sh` 和 `configurations/`。

资源只在每个任务目录的 `tuxiv.conf` → `job.general` 中设置。提交、状态、日志、下载、停止统一使用 SDK。SDK 内部使用 SSH 协议和 Slurm；学生不登录平台服务器，也不自行建立 SSH 转发。

官方依据：[Quickstart 提交作业](https://github.com/turingaicloud/quickstart#submit-a-job)、[SDK Job Configuration](https://github.com/turingaicloud/tcloud-sdk#job-configuration)、[FAQ](https://github.com/turingaicloud/quickstart/blob/master/FAQ.md)。

## 演示时再次启动

直接在 `qwen-public-serve` 目录执行 `tcloud submit`，不需要安装依赖或下载模型。服务复用公共权重，编译缓存保存到本人 USERDIR 的持久目录；同一版本和硬件可复用有效缓存。模型加载、进程初始化和 GPU 预热仍需执行，具体启动时间以实测为准。服务代码和运行依赖不会自动升级。
