"""M-SAgent V1 新内核包。

本包只承载新架构 V1 的代码骨架，严格按照 docs/ 下的架构文档组织：

- `core` 放状态对象、公共 contract、策略和配置；
- `orchestrator` 放唯一主控制点；
- `modules` 放核心业务模块；
- `infra` 放外部模型与存储适配器；
- `service` 放 CLI / API 等薄入口。
"""

