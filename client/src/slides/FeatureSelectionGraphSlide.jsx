export default function FeatureSelectionGraphSlide() {
  const chartData = [
    { stage: 'Raw Features', count: 86, color: 'bg-danger/80' },
    { stage: 'Pre-TSFresh Filter', count: 50, color: 'bg-warning/80' },
    { stage: 'TSFresh Extractions', count: 1200, color: 'bg-primary/50' },
    { stage: 'P-Value (< 0.05)', count: 320, color: 'bg-primary/70' },
    { stage: 'Collinearity (|r| < 0.9)', count: 154, color: 'bg-primary/90' },
    { stage: 'High IC Targets', count: 42, color: 'bg-success/80' },
  ]

  const maxCount = 1200;

  return (
    <div className="flex items-center justify-center h-full px-8 pb-10">
      <div className="max-w-6xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">05.3 / Feature Engineering III</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Mining the Signal: 50+ to core features</span>
        </h2>
        
        <p className="text-text-secondary mb-8 fade-up opacity-0 stagger-2 max-w-3xl">
          We extract massive dimensionality, but aggressively prune using strict quantitative gates (Information Coefficient, Hypothesis Testing) to avoid the curse of dimensionality.
        </p>

        <div className="glass-card p-8 fade-up opacity-0 stagger-3">
          <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider mb-6">Feature Pruning Funnel</h3>
          
          <div className="space-y-6">
            {chartData.map((d, index) => {
              const widthPerc = Math.max((d.count / maxCount) * 100, 2); // At least 2% to show a small bar
              return (
                <div key={d.stage} className="flex flex-col md:flex-row md:items-center gap-2 md:gap-4">
                  <div className="w-48 text-sm font-medium text-text-secondary md:text-right shrink-0">
                    {d.stage}
                  </div>
                  
                  <div className="flex-1 flex items-center gap-3">
                    <div className="flex-1 h-6 bg-surface-lighter/50 rounded-full overflow-hidden relative">
                      <div 
                        className={`absolute top-0 left-0 h-full rounded-full transition-all duration-1000 ease-out ${d.color}`}
                        style={{ width: `${widthPerc}%`, animationDelay: `${index * 150}ms` }}
                      />
                    </div>
                    <div className="w-12 text-sm font-mono font-bold text-accent text-right shrink-0">
                      {d.count}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-8 pt-6 border-t border-border/30 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-xl font-bold text-primary-light">42</div>
              <div className="text-xs text-text-muted mt-1">Final Features</div>
            </div>
            <div>
              <div className="text-xl font-bold text-success">96.5%</div>
              <div className="text-xs text-text-muted mt-1">Noise Reduced</div>
            </div>
            <div>
              <div className="text-xl font-bold text-accent">IC &gt; 0.02</div>
              <div className="text-xs text-text-muted mt-1">Minimum Signal Rate</div>
            </div>
            <div>
              <div className="text-xl font-bold text-warning">0 Overlap</div>
              <div className="text-xs text-text-muted mt-1">Collinear Drop</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
