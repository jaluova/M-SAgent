import { type ReactNode } from 'react'

interface MainLayoutProps {
  readonly leftColumn: ReactNode
  readonly centerColumn: ReactNode
  readonly rightColumn: ReactNode
  readonly bottomSection: ReactNode
}

export function MainLayout(props: MainLayoutProps) {
  const { leftColumn, centerColumn, rightColumn, bottomSection } = props

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__title">
          <h1>M-SAgent</h1>
        </div>
      </header>

      <main className="workspace">
        <section className="workspace-main">
          <div className="stack">{leftColumn}</div>
          <div className="stack">{centerColumn}</div>
          <div className="stack">{rightColumn}</div>
        </section>
        <section className="workspace-bottom">{bottomSection}</section>
      </main>
    </div>
  )
}
