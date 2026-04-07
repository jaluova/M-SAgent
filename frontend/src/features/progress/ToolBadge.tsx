import { type ToolName } from '../../types/job'

const toolLabels: Record<ToolName, string> = {
  object_locator: '目标定位',
  concept_generator: '概念生成',
  image_enhancer: '图像增强',
  report_no_mask: '未找到目标',
}

interface ToolBadgeProps {
  readonly tool: ToolName | null
}

export function ToolBadge(props: ToolBadgeProps) {
  const { tool } = props

  return <span className="tool-badge">{tool ? toolLabels[tool] : '待定'}</span>
}
