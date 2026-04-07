import { type ToolName } from '../../types/job'

const toolLabels: Record<ToolName, string> = {
  object_locator: 'Object Locator',
  concept_generator: 'Concept Generator',
  image_enhancer: 'Image Enhancer',
  report_no_mask: 'Report No Mask',
}

interface ToolBadgeProps {
  readonly tool: ToolName | null
}

export function ToolBadge(props: ToolBadgeProps) {
  const { tool } = props

  return (
    <span className="tool-badge">{tool ? toolLabels[tool] : '等待工具选择'}</span>
  )
}
