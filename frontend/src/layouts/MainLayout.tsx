import { type ReactNode } from 'react'

interface MainLayoutProps {
  readonly leftColumn: ReactNode
  readonly rightColumn: ReactNode
}

export function MainLayout(props: MainLayoutProps) {
  const { leftColumn, rightColumn } = props

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__title">
          <p className="eyebrow">M-SAgent</p>
          <h1>Referring Segmentation</h1>
        </div>
      </header>

      <main className="app-grid">
        <div className="stack">{leftColumn}</div>
        <div className="stack">{rightColumn}</div>
      </main>
    </div>
  )
}
