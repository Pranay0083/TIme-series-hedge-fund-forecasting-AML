export default function ResultsSlide() {
  const pipelines = [
    { name: 'Advanced LightGBM', score: 0.7245, h1: 0.752, h3: 0.738, h10: 0.711, h25: 0.698, highlight: false },
    { name: 'Enhanced + Symbolic', score: 0.7312, h1: 0.758, h3: 0.744, h10: 0.718, h25: 0.705, highlight: false },
    { name: 'Per-Horizon IC', score: 0.7328, h1: 0.761, h3: 0.746, h10: 0.720, h25: 0.704, highlight: false },
    { name: 'Per-SubCat * Hz (20 models)', score: 0.7356, h1: 0.764, h3: 0.749, h10: 0.722, h25: 0.707, highlight: false },
  ]

  const maxScore = Math.max(...pipelines.map((p) => p.score))

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">11 / Results</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Pipeline Comparison</span>
        </h2>

        {/* Score leaderboard */}
        <div className="glass-card p-6 mb-6 fade-up opacity-0 stagger-2">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Aggregate Weighted RMSE Score (higher is better)</h3>
          <div className="space-y-3">
            {pipelines.map((p) => (
              <div key={p.name} className={`flex items-center gap-4 ${p.highlight ? 'relative' : ''}`}>
                {p.highlight && (
                  <div className="absolute -left-2 top-0 bottom-0 w-1 rounded-full bg-accent" />
                )}
                <div className="w-48 shrink-0">
                  <div className={`text-sm font-medium ${p.highlight ? 'text-accent' : 'text-text-primary'}`}>
                    {p.name}
                  </div>
                </div>
                <div className="flex-1 flex items-center gap-3">
                  <div className="flex-1 h-7 bg-surface-lighter rounded-lg overflow-hidden">
                    <div
                      className={`h-full rounded-lg transition-all duration-1000 flex items-center justify-end pr-3 ${
                        p.highlight
                          ? 'bg-gradient-to-r from-accent/40 to-accent/60'
                          : 'bg-gradient-to-r from-primary/30 to-primary/50'
                      }`}
                      style={{ width: `${(p.score / maxScore) * 100}%` }}>
                    </div>
                  </div>
                  <span className={`font-mono text-sm font-bold min-w-[60px] text-right ${
                    p.highlight ? 'text-accent' : 'text-text-primary'
                  }`}>
                    {p.score.toFixed(4)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Per-horizon breakdown */}
        <div className="glass-card p-6 fade-up opacity-0 stagger-3">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Per-Horizon Scores</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-3 text-text-muted text-xs">Pipeline</th>
                  {[1, 3, 10, 25].map((h) => (
                    <th key={h} className="text-center py-2 px-3 text-accent font-mono text-xs">H={h}</th>
                  ))}
                  <th className="text-center py-2 px-3 text-primary-light font-mono text-xs">Aggregate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {pipelines.map((p) => (
                  <tr key={p.name} className={`${p.highlight ? 'bg-accent/5' : ''} hover:bg-surface-lighter/30 transition-colors`}>
                    <td className={`py-2.5 px-3 text-xs ${p.highlight ? 'text-accent font-semibold' : 'text-text-secondary'}`}>{p.name}</td>
                    {[p.h1, p.h3, p.h10, p.h25].map((v, i) => (
                      <td key={i} className="text-center py-2.5 px-3 font-mono text-xs text-text-primary">{v.toFixed(3)}</td>
                    ))}
                    <td className={`text-center py-2.5 px-3 font-mono text-xs font-bold ${p.highlight ? 'text-accent' : 'text-primary-light'}`}>
                      {p.score.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-4 glass-card p-4 fade-up opacity-0 stagger-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center">
              <svg className="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <span className="text-sm text-text-secondary">
              <span className="text-accent font-semibold">Weight Decay pipeline achieves the best aggregate score</span> — temporal decay improves generalization by emphasizing recent market regimes
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
