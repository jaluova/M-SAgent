"""底层模型运行时实现。

当前阶段对外稳定承诺的只有 `TrainAdapterRuntime` 抽象。
其他 request / prediction / feature-context DTO 仅供 infra/runtime 内部协作使用。
"""

from msagent.infra.runtime.train_adapter_runtime import TrainAdapterRuntime

__all__ = ["TrainAdapterRuntime"]
