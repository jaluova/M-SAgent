import { type ToolName } from '../../types/job'

const toolLabels: Record<ToolName, string> = {
  object_locator: 'Locator',
  concept_generator: 'Concept',
  image_enhancer: 'Enhancer',
  report_no_mask: 'No Mask',
}

interface ToolBadgeProps {
  readonly tool: ToolName | null
}

export function ToolBadge(props: ToolBadgeProps) {
  const { tool } = props

  return <span className="tool-badge">{tool ? toolLabels[tool] : '待定'}</span>
}
