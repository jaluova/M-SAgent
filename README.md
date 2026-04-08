# M-SAgent

M-SAgent 是一个面向指代表达图像分割的多模块系统。它把 `Qwen2.5-VL` 的多模态理解能力、`GridGround` 的目标定位能力和 `SAM3` 的精细分割能力串成一个可迭代的 Agent 流程，用来从自然语言描述中找到目标并输出掩码。

这个仓库已经整理成可直接运行的推理与部署版本，包含：

- 命令行单图推理入口
- FastAPI 后端
- React + Vite 前端界面
- 内嵌式 GridGround 运行时
- 可选的外部 TrainAdapter 定位服务接入

## 适用场景

- 根据中文或英文指代表达对图像中的目标做分割
- 需要查看中间推理过程，而不是只拿最终掩码
- 单机部署一个可上传图片、查看迭代日志和下载结果的 Web 服务
- 在研究或比赛环境中快速验证“定位 + 分割 + 自评估”流程

## 系统流程

一次完整推理大致如下：

1. `MLLMProcessor` 载入 Qwen2.5-VL，读取原图、网格图和历史结果，决定下一步该调用什么工具。
2. `ObjectLocator`、`ConceptGenerator`、`ImageEnhancer` 三类工具之一被触发。
3. 工具把候选提示交给 `SAMProcessor`，由 SAM3 生成一个或多个候选掩码。
4. 管道生成可视化检查图，再由 MLLM 对当前分割结果做接受或拒绝判断。
5. 若结果被拒绝，系统保留合适候选并继续下一轮；若接受，则输出最终结果图和掩码。

默认最多迭代 5 轮。

## 核心特性

- `Qwen2.5-VL + SAM3` 联合推理，而不是单模型直接出结果
- 支持 `object_locator`、`concept_generator`、`image_enhancer`、`report_no_mask` 四类动作
- 支持中间检查图、放大 review 图、迭代日志和最终叠加图
- Web 端支持任务排队、WebSocket 进度事件、结果预览和掩码下载
- 单 GPU 稳定模式默认使用内嵌 GridGround，避免再加载第二份 Qwen 模型

## 项目结构

```text
M-SAgent/
├── README.md
├── config.py                    # 全局配置与环境变量解析
├── main.py                      # CLI 入口
├── pipeline.py                  # 主推理管道
├── mllm_processor.py            # Qwen2.5-VL 决策与评估
├── sam_processor.py             # SAM3 分割封装
├── tools/                       # 定位、概念生成、图像增强、TrainAdapter 客户端
├── server/                      # FastAPI 服务与任务管理
├── frontend/                    # React + Vite 前端
├── gridground_runtime/          # 内嵌 GridGround 运行时
├── prompts/                     # 系统提示词
├── scripts/                     # 启动、下载与部署脚本
├── utils/                       # 图像、可视化、定位辅助工具
└── example/                     # 示例图片
```

## 环境要求

- Python `3.10`
- Node.js `18+`，仅前端开发或重新构建前端时需要
- 建议使用 CUDA 环境运行
- 需要本地准备 Qwen2.5-VL、SAM3 checkpoint 和 GridGround 权重

## 安装依赖

推荐使用 Conda：

```bash
conda create -n m_sagent python=3.10 -y
conda activate m_sagent
```

先安装与你的 CUDA 版本匹配的 PyTorch。下面是 CUDA 12.4 示例：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

再安装其余依赖：

```bash
pip install -r requirements.txt
```

`SAM3` 代码不再作为仓库内置依赖分发。运行前请单独准备一份外部 `SAM3` 代码目录，并在启动前设置 `M_SAGENT_SAM3_MODEL_PATH`。

如果要开发或重建前端：

```bash
cd frontend
npm install
```

## 模型与关键环境变量

最少需要确认下面这些路径：

```bash
export M_SAGENT_PYTHON=/path/to/python
export M_SAGENT_QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export M_SAGENT_SAM3_MODEL_PATH=/path/to/external/sam3
export M_SAGENT_SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
export GRIDGROUND_MODEL_DIR=/path/to/GridGround-TextGuided
```

常用环境变量说明：

| 变量 | 作用 |
| --- | --- |
| `M_SAGENT_PYTHON` | 启动脚本使用的 Python 解释器 |
| `M_SAGENT_QWEN_MODEL_PATH` | Qwen2.5-VL 模型目录 |
| `M_SAGENT_SAM3_MODEL_PATH` | 外部 `SAM3` 代码目录 |
| `M_SAGENT_SAM3_CHECKPOINT_PATH` | SAM3 权重文件 |
| `GRIDGROUND_MODEL_DIR` | GridGround 模型目录 |
| `GRIDGROUND_BACKEND` | 定位后端，通常为 `embedded` 或 `http` |
| `TRAIN_ADAPTER_ENABLED` | 是否启用外部 TrainAdapter 服务 |
| `TRAIN_ADAPTER_URL` | 外部定位服务地址，默认 `http://127.0.0.1:8765` |
| `M_SAGENT_SERVER_HOST` | Web 服务监听地址 |
| `M_SAGENT_SERVER_PORT` | Web 服务端口，默认 `8000` |

## 源码提交说明

比赛源码包建议仅包含本仓库中的团队自研代码与必要工程文件，不包含第三方 `SAM3` 源码、前端构建产物、缓存目录和本地版本控制元数据。开发环境可以保留本地 `sam3/` 目录，但提交压缩包时建议排除 `sam3/`、`frontend/dist/`、`__pycache__/`、`.git/` 和 `.DS_Store`。

如果本地还没有 GridGround 权重，可以用脚本下载：

```bash
python scripts/download_gridground_model.py
```

## 快速开始

### 1. 命令行单图测试

```bash
./scripts/run_single_gpu_stable.sh --image example/truck.jpg --text "truck"
```

常用参数：

- `--image`：输入图片路径
- `--text`：目标描述
- `--max_iter`：最大迭代次数，默认 `5`
- `--output_dir`：输出目录

CLI 成功后通常会在 `outputs/` 下生成：

- `final_result_*.jpg`：最终可视化结果
- `final_mask_*.npy`：最终掩码数组
- `iter*_*.jpg`：中间检查图

### 2. 启动 Web 服务

如果只是直接使用仓库自带前端，`frontend/dist` 已经存在；若你修改过前端源码，先重新构建：

```bash
cd frontend
npm install
npm run build
cd ..
```

然后启动服务：

```bash
./scripts/start_web_single_gpu_stable.sh
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

浏览器访问：

```text
http://127.0.0.1:8000
```

这个脚本对应的是单 GPU 稳定模式，特点是：

- 固定使用 `GRIDGROUND_BACKEND=embedded`
- 默认关闭 `TRAIN_ADAPTER_ENABLED`
- 强制只启一个 uvicorn worker
- 启动前会检查 SAM3 checkpoint、GridGround 目录和 GPU 空闲显存

### 3. 前端本地开发

```bash
cd frontend
npm run dev
```

默认开发地址通常为：

```text
http://127.0.0.1:5173
```

前端开发时会访问后端 `/api/*` 接口；后端允许默认的本地开发跨域来源。

## Web 端交互说明

前端页面主要分成四块：

- 左侧：上传图片、填写指代表达、设置最大迭代次数
- 中间：展示每一轮工具选择和中间检查图
- 右侧：展示事件流和日志
- 下方：展示最终掩码叠加结果，并提供结果图、PNG 掩码、NPY 掩码下载

后端会把每个上传任务保存到独立目录，通常位于：

```text
remote_artifacts/server_jobs/<job_id>/
```

目录里常见产物包括：

- `result.png`
- `result_preview.jpg`
- `mask.png`
- `mask.npy`

## API 概览

主要接口如下：

- `GET /api/health`：服务健康状态
- `POST /api/jobs`：上传图片并创建任务
- `GET /api/jobs/{job_id}`：查询任务状态
- `DELETE /api/jobs/{job_id}`：删除任务与产物
- `GET /api/jobs/{job_id}/result`：下载结果图
- `GET /api/jobs/{job_id}/mask?format=png|npy`：下载掩码
- `WS /api/jobs/{job_id}/ws`：订阅任务事件

前端消费的关键事件包括：

- `queued`
- `started`
- `iteration_start`
- `tool_selected`
- `segmentation_result`
- `evaluation`
- `complete`
- `error`

## 关于 GridGround 与 TrainAdapter

仓库支持两种定位方式：

### 1. 内嵌 GridGround

这是当前默认推荐方式，尤其适合单 GPU 部署。

特点：

- 直接在本进程内复用已加载的 Qwen backbone
- 避免同时启动第二个 Qwen 服务导致显存爆炸
- `start_web_single_gpu_stable.sh` 就是按这种方式设计的

### 2. 外部 TrainAdapter 服务

如果你希望把定位模块拆成单独服务，可以先启动 CPU 版服务：

```bash
./scripts/start_inference_service_cpu.sh
```

它最终会调用 `scripts/start_train_adapter_service.sh`。

如果要让主程序通过 HTTP 调这个服务，至少需要设置：

```bash
export GRIDGROUND_BACKEND=http
export TRAIN_ADAPTER_ENABLED=true
export TRAIN_ADAPTER_URL=http://127.0.0.1:8765
```

注意：仓库自带的 `start_web_single_gpu_stable.sh` 会拒绝在 Web 单 GPU 模式下使用 `http` 后端，因为那样通常意味着会再加载一份 Qwen，容易 OOM。

## 常见目录与文件说明

- [main.py](main.py)：命令行入口
- [pipeline.py](pipeline.py)：核心迭代逻辑
- [config.py](config.py)：路径、模型、服务和阈值配置
- [server/app.py](server/app.py)：FastAPI 应用入口
- [server/job_manager.py](server/job_manager.py)：队列和任务产物管理
- [frontend/src/App.tsx](frontend/src/App.tsx)：前端主界面
- [scripts/start_web_single_gpu_stable.sh](scripts/start_web_single_gpu_stable.sh)：推荐的 Web 启动脚本
- [scripts/run_single_gpu_stable.sh](scripts/run_single_gpu_stable.sh)：推荐的 CLI 启动脚本

## 常见问题

### 1. 启动时报模型路径不存在

优先检查以下变量是否正确：

- `M_SAGENT_QWEN_MODEL_PATH`
- `M_SAGENT_SAM3_CHECKPOINT_PATH`
- `GRIDGROUND_MODEL_DIR`

### 2. Web 服务启动时直接退出

`scripts/start_web_single_gpu_stable.sh` 会在启动前做多项保护检查，包括：

- Python 解释器是否存在
- SAM3 checkpoint 是否存在
- GridGround 权重目录是否存在
- 是否错误地配置成 `GRIDGROUND_BACKEND=http`
- GPU 空闲显存是否低于 `M_SAGENT_MIN_FREE_GPU_MB`

### 3. 前端能打开，但没有结果

可以按这个顺序排查：

1. 先访问 `/api/health`
2. 查看 `uvicorn.log`
3. 查看任务目录下是否生成了中间图和掩码
4. 确认上传的是图片文件，且大小没有超过后端限制

## 备注

- `frontend/dist` 是 FastAPI 直接托管的生产前端产物
- `frontend/src` 保留用于继续开发
- 默认配置更偏向“单机、单 GPU、稳定优先”的部署方式
