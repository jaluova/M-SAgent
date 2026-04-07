# M-SAgent 前端架构方案

## Context

M-SAgent 是一个基于 Qwen2.5-VL + SAM3 的智能体式图像分割系统，当前仅有 CLI 入口。用户输入图片+自然语言描述，系统通过最多 5 轮迭代的 Agent Loop（MLLM 选工具 → 分割 → 评估）产出精确的像素级分割掩码。处理耗时 30s-2min（A100 GPU）。需要一个 Web 前端来提供交互式的图片上传、实时进度展示和结果可视化。

---

## 整体架构

```
┌─────────────┐    REST/WS     ┌───────────────┐    Python调用    ┌──────────────────┐
│   React SPA │ ◄────────────► │  FastAPI 服务  │ ──────────────► │ MLLMSAMPipeline  │
│  (Vite+TS)  │    :5173       │   (Uvicorn)   │    :8000        │  (现有pipeline)  │
└─────────────┘                └───────────────┘                  └──────────────────┘
```

- **REST** 用于：上传图片、查询状态、下载结果
- **WebSocket** 用于：实时推送 Agent Loop 每一步的进度事件

---

## 技术栈

| 层级 | 选型 | 理由 |
|------|------|------|
| 前端框架 | React 18 + TypeScript | 生态成熟，ML 研究者易上手 |
| 构建工具 | Vite | 快速开发，零配置 |
| 样式 | Tailwind CSS | 快速原型，无需写自定义 CSS |
| 状态管理 | Zustand | 1KB，支持 selector 精细更新，比 Redux 轻量 |
| 后端 API | FastAPI + Uvicorn | 原生 async，内置 WebSocket 支持 |
| 通信 | WebSocket（进度流）+ REST（CRUD） | WS 可双向通信，未来支持取消/调参 |

---

## 一、后端 API 层（新增 `server/` 目录）

### 1. 对现有 Pipeline 的最小改动

在 `pipeline.py` 的 `run()` 方法中增加可选的 `on_progress` 回调参数，在关键节点发送事件：

| 触发点 | 事件类型 | 携带数据 |
|--------|---------|---------|
| 迭代开始 | `iteration_start` | iteration, max_iterations |
| MLLM 选择工具 | `tool_selected` | tool, params |
| 工具产出分割结果 | `segmentation_result` | tool, score, check_image_url |
| MLLM 评估结果 | `evaluation` | verdict, rejected_indices |
| 完成 | `complete` | success, best_score, iterations, mask_count |
| 出错 | `error` | message |

不传回调时，pipeline 行为完全不变（CLI 兼容）。

### 2. REST 端点

```
POST   /api/jobs                         — 创建分割任务（multipart: image + text + max_iter）
GET    /api/jobs/{job_id}                — 查询任务状态
GET    /api/jobs/{job_id}/result         — 获取最终结果图（JPEG）
GET    /api/jobs/{job_id}/mask           — 下载掩码文件（.npy / .png）
GET    /api/jobs/{job_id}/images/{name}  — 获取中间结果图
DELETE /api/jobs/{job_id}                — 取消/清理任务
GET    /api/health                       — 健康检查 + GPU 状态
```

### 3. WebSocket 端点

```
WS /api/jobs/{job_id}/ws  — 订阅任务进度事件流
```

### 4. 任务队列

- GPU 单例，内存中维护任务队列（`asyncio.Queue`）
- 单 worker 协程，逐一处理任务
- Pipeline 在线程池 (`asyncio.to_thread`) 中运行，回调通过 `asyncio.Queue` 桥接到 async WebSocket

### 5. 文件结构

```
server/
  app.py                — FastAPI 应用工厂、CORS、生命周期
  config.py             — 服务端配置（端口、CORS、上传限制）
  models.py             — Pydantic 模型（JobCreate, JobStatus, ProgressEvent）
  job_manager.py        — 任务队列、worker、内存存储
  pipeline_adapter.py   — 包装 MLLMSAMPipeline，注入 on_progress 回调
  routes/
    jobs.py             — 任务 CRUD 路由
    ws.py               — WebSocket 路由
    health.py           — 健康检查路由
```

---

## 二、前端项目结构

```
frontend/
  index.html
  package.json
  vite.config.ts
  tailwind.config.ts
  tsconfig.json
  src/
    main.tsx                            — 入口
    App.tsx                             — 根组件

    api/
      client.ts                         — fetch 封装、base URL
      jobs.ts                           — REST 调用（createJob, getJob, getHealth）
      websocket.ts                      — WebSocket 连接管理

    types/
      job.ts                            — TS 类型定义

    stores/
      job-store.ts                      — Zustand store

    features/
      upload/
        UploadPanel.tsx                 — 图片上传 + 文字输入 + 提交
        ImagePreview.tsx                — 上传图片预览
      progress/
        ProgressPanel.tsx               — 实时进度容器
        IterationTimeline.tsx           — 迭代时间线可视化
        ToolBadge.tsx                   — 工具名称徽章
        AgentLogStream.tsx              — 滚动事件日志
      result/
        ResultPanel.tsx                 — 最终结果容器
        MaskOverlay.tsx                 — 掩码叠加展示 + 透明度滑块
        ResultStats.tsx                 — 分数、迭代次数、掩码数量
        DownloadActions.tsx             — 下载掩码/结果图
      shared/
        StatusIndicator.tsx             — 状态徽章
        ImageViewer.tsx                 — 可缩放图片组件
        ScoreBadge.tsx                  — 置信度分数展示

    hooks/
      useJob.ts                         — 组合 store + WebSocket 生命周期

    layouts/
      MainLayout.tsx                    — 页面布局（头部 + 双栏主体）
```

---

## 三、组件层级

```
App
  MainLayout
    Header（标题 + 健康状态指示）

    左栏:
      UploadPanel
        ImagePreview（拖拽上传/点击上传）
        文字输入框（referring expression）
        最大迭代次数滑块
        提交按钮

    右栏（条件渲染）:
      StatusIndicator（排队中/运行中/已完成/失败）

      [运行中或已完成时]:
      ProgressPanel
        IterationTimeline（每轮迭代的工具+分数+判定）
        AgentLogStream（实时滚动日志）

      [已完成时]:
      ResultPanel
        MaskOverlay（原图+掩码叠加，支持透明度调节）
        ResultStats（最终分数、轮次、掩码数）
        DownloadActions（下载 .npy / .png / 结果图）
```

---

## 四、核心 TypeScript 类型

```typescript
type ToolName = 'object_locator' | 'concept_generator' | 'image_enhancer' | 'report_no_mask';

// WebSocket 事件（discriminated union）
type ProgressEvent =
  | { readonly type: 'queued'; readonly position: number }
  | { readonly type: 'started' }
  | { readonly type: 'iteration_start'; readonly iteration: number; readonly maxIterations: number }
  | { readonly type: 'tool_selected'; readonly iteration: number; readonly tool: ToolName; readonly params: Record<string, unknown> }
  | { readonly type: 'segmentation_result'; readonly iteration: number; readonly tool: ToolName; readonly score: number; readonly checkImageUrl: string }
  | { readonly type: 'evaluation'; readonly iteration: number; readonly verdict: 'Accept' | 'Reject'; readonly rejectedIndices: readonly number[] }
  | { readonly type: 'complete'; readonly result: JobResult }
  | { readonly type: 'error'; readonly message: string };

interface JobResult {
  readonly success: boolean;
  readonly bestScore: number;
  readonly iterations: number;
  readonly maskCount: number;
  readonly resultImageUrl: string;
  readonly maskUrl: string;
}

interface IterationSnapshot {
  readonly iteration: number;
  readonly tool: ToolName;
  readonly score: number | null;
  readonly verdict: 'Accept' | 'Reject' | 'pending';
  readonly checkImageUrl: string | null;
}
```

---

## 五、状态管理（Zustand Store）

单一 store，所有状态更新由 WebSocket 事件驱动：

```typescript
interface JobStore {
  readonly jobStatus: 'idle' | 'uploading' | 'queued' | 'running' | 'complete' | 'failed';
  readonly jobId: string | null;
  readonly textQuery: string;
  readonly maxIterations: number;
  readonly events: ReadonlyArray<ProgressEvent>;
  readonly currentIteration: number;
  readonly currentTool: string | null;
  readonly iterations: ReadonlyArray<IterationSnapshot>;
  readonly result: JobResult | null;

  // Actions
  setTextQuery: (q: string) => void;
  setMaxIterations: (n: number) => void;
  submitJob: (image: File) => Promise<void>;
  handleEvent: (event: ProgressEvent) => void;
  reset: () => void;
}
```

---

## 六、实时通信流程

1. 用户点击提交 → `POST /api/jobs` 上传图片+文字 → 返回 `job_id`
2. 前端立即建立 `WS /api/jobs/{job_id}/ws` 连接
3. 后端每个 pipeline 步骤通过 `on_progress` 回调发送事件
4. 前端 `handleEvent` 更新 store → 组件自动重渲染
5. 中间结果图通过 URL 引用（`<img src="/api/jobs/{id}/images/...">`），不在 WS 中传 base64
6. 收到 `complete` 事件后关闭 WS

---

## 七、实施阶段

### Phase 1: 后端 API（优先）
1. 在 `pipeline.py` 的 `run()` 中增加 `on_progress` 回调钩子
2. 创建 `server/` 目录，实现 FastAPI 应用、任务管理器、pipeline 适配器
3. 用 wscat 或简单 HTML 验证 WebSocket 事件流

### Phase 2: 前端脚手架
4. 初始化 Vite + React + TS + Tailwind 项目
5. 实现 `api/` 层（REST client, WebSocket manager）
6. 实现 Zustand store 和类型定义

### Phase 3: UI 组件
7. UploadPanel（拖拽上传、文字输入、提交）
8. ProgressPanel + IterationTimeline + AgentLogStream
9. ResultPanel + MaskOverlay + 下载功能

### Phase 4: 完善
10. 错误处理、断线重连、Loading 状态
11. 移动端适配
12. Header 健康检查展示

---

## 八、关键文件（需修改/参考）

| 文件 | 操作 | 说明 |
|------|------|------|
| `pipeline.py` | **修改** | 添加 `on_progress` 回调钩子 |
| `config.py` | **修改** | 添加服务端配置项 |
| `main.py` | 参考 | 理解入口逻辑 |
| `mllm_processor.py` | 参考 | 理解事件触发时机 |
| `sam_processor.py` | 参考 | 理解 GPU 内存管理 |

---

## 九、验证方式

1. 启动 FastAPI：`uvicorn server.app:app --host 0.0.0.0 --port 8000`
2. 启动前端：`cd frontend && npm run dev`
3. 上传测试图片（`example/` 目录中的样本）+ 输入描述文字
4. 验证 WebSocket 实时推送每一步的进度事件
5. 验证最终结果图和掩码可正确下载
6. 验证 GPU 单例队列：同时提交两个任务，第二个应排队等待
