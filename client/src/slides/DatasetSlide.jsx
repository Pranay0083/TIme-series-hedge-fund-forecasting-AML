export default function DatasetSlide() {
  const columns = [
    { name: 'id', type: 'int', desc: 'Unique sample identifier' },
    { name: 'code', type: 'cat', desc: 'Asset code identifier' },
    { name: 'sub_code', type: 'cat', desc: 'Sub-asset classification' },
    { name: 'sub_category', type: 'cat', desc: '5 asset sub-categories' },
    { name: 'horizon', type: 'int', desc: '1, 3, 10, or 25 days' },
    { name: 'ts_index', type: 'int', desc: 'Temporal index (ordering)' },
    { name: 'feature_a..ch', type: 'float', desc: '90+ anonymized numeric features' },
    { name: 'y_target', type: 'float', desc: 'Return to predict' },
    { name: 'weight', type: 'float', desc: 'Sample importance for scoring' },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">03 / Dataset</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Data Structure</span>
        </h2>

        <div className="grid lg:grid-cols-5 gap-6">
          {/* Stats cards */}
          <div className="lg:col-span-2 space-y-4 fade-up opacity-0 stagger-2">
            {[
              { label: 'Features', value: '90+', sub: 'Anonymized signals' },
              { label: 'Horizons', value: '4', sub: '1, 3, 10, 25 days' },
              { label: 'Sub-categories', value: '5', sub: 'Asset classes' },
              { label: 'Target', value: 'y_target', sub: 'Continuous returns' },
            ].map((stat) => (
              <div key={stat.label} className="glass-card p-4 flex items-center gap-4">
                <div className="text-3xl font-bold font-mono gradient-text min-w-[60px]">{stat.value}</div>
                <div>
                  <div className="text-sm font-semibold text-text-primary">{stat.label}</div>
                  <div className="text-xs text-text-muted">{stat.sub}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Schema table */}
          <div className="lg:col-span-3 glass-card p-1 fade-up opacity-0 stagger-3">
            <div className="px-4 py-3 border-b border-border">
              <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                <svg className="w-4 h-4 text-primary-light" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7C5 4 4 5 4 7z" />
                </svg>
                Schema Overview
              </h3>
            </div>
            <div className="divide-y divide-border/50">
              {columns.map((col) => (
                <div key={col.name} className="px-4 py-2.5 flex items-center gap-3 hover:bg-surface-lighter/50 transition-colors">
                  <code className="text-xs font-mono text-accent min-w-[120px]">{col.name}</code>
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                    col.type === 'float' ? 'bg-primary/10 text-primary-light' :
                    col.type === 'cat' ? 'bg-success/10 text-success' :
                    'bg-warning/10 text-warning'
                  }`}>{col.type}</span>
                  <span className="text-xs text-text-muted">{col.desc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Competition metric */}
        <div className="mt-6 glass-card p-5 fade-up opacity-0 stagger-4">
          <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
            <svg className="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
            Competition Metric — Weighted RMSE Score
          </h3>
          <div className="bg-surface rounded-lg p-4 font-mono text-sm text-center">
            <span className="text-text-primary">Score = </span>
            <span className="text-primary-light">sqrt</span>
            <span className="text-text-muted">(</span>
            <span className="text-text-primary">1 - clip</span>
            <span className="text-text-muted">(</span>
            <span className="text-accent">Sum(w * (y - pred)²)</span>
            <span className="text-text-muted"> / </span>
            <span className="text-accent">Sum(w * y²)</span>
            <span className="text-text-muted">)</span>
            <span className="text-text-muted">)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
