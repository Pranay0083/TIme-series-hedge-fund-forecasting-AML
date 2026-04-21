export default function MLPArchitectureSlide() {
  const layers = [
    { name: 'Embeddings', dim: 'Σ emb_dim', color: 'bg-accent/30', textColor: 'text-accent', desc: 'per-categorical' },
    { name: 'Concat', dim: 'emb + num', color: 'bg-primary/20', textColor: 'text-primary-light', desc: 'merge layer' },
    { name: 'Linear → ReLU → BN', dim: '512', color: 'bg-primary/30', textColor: 'text-primary-light', desc: 'block 1' },
    { name: 'Dropout(0.25)', dim: '512', color: 'bg-warning/20', textColor: 'text-warning', desc: 'regularize' },
    { name: 'Linear → ReLU', dim: '256', color: 'bg-primary/30', textColor: 'text-primary-light', desc: 'block 2' },
    { name: 'Dropout(0.25)', dim: '256', color: 'bg-warning/20', textColor: 'text-warning', desc: 'regularize' },
    { name: 'Linear → ReLU', dim: '128', color: 'bg-primary/30', textColor: 'text-primary-light', desc: 'block 3' },
    { name: 'Dropout(0.25)', dim: '128', color: 'bg-warning/20', textColor: 'text-warning', desc: 'regularize' },
    { name: 'Linear → Output', dim: '1', color: 'bg-success/30', textColor: 'text-success', desc: 'prediction' },
  ]

  const config = [
    { param: 'batch_size', value: '2,048', why: 'MLP is fast — large batches stabilize gradient estimates, reduce noise' },
    { param: 'learning_rate', value: '1e-3', why: 'Standard for Adam-family optimizers — too low wastes epochs, too high diverges' },
    { param: 'weight_decay', value: '1e-5', why: 'L2 regularization — prevents large weights without being so strong it kills capacity' },
    { param: 'epochs', value: '10', why: 'MLP converges fast on tabular data — longer risks memorizing noisy targets' },
    { param: 'dropout', value: '0.25', why: 'Tabular data needs moderate dropout — too high kills signal, too low overfits' },
    { param: 'grad_clip', value: '1.0', why: 'Prevents gradient explosion from fat-tailed target distribution (kurtosis ~290)' },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">Phase 2 / Architecture</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">MLP + Embeddings</span>
        </h2>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Architecture diagram */}
          <div className="fade-up opacity-0 stagger-2">
            <div className="glass-card p-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
                <svg className="w-4 h-4 text-primary-light" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                Network Architecture
              </h3>

              {/* Layer visualization */}
              <div className="space-y-1.5">
                {layers.map((layer, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className={`flex-1 h-7 rounded-lg ${layer.color} flex items-center justify-between px-3`}>
                      <span className={`text-[10px] font-mono ${layer.textColor}`}>{layer.name}</span>
                      <span className={`text-[10px] font-mono ${layer.textColor} opacity-70`}>{layer.dim}</span>
                    </div>
                    <span className="text-[9px] text-text-muted w-16 text-right">{layer.desc}</span>
                  </div>
                ))}
              </div>

              {/* Embedding formula */}
              <div className="mt-4 bg-surface rounded-lg p-3">
                <div className="text-[10px] text-text-muted uppercase tracking-wider mb-2">Embedding Dimension Rule</div>
                <div className="font-mono text-xs text-text-secondary">
                  emb_dim = <span className="text-accent">min</span>(50, cardinality <span className="text-accent">//</span> 2 + 1)
                </div>
                <div className="text-[10px] text-text-muted mt-1">
                  Caps at 50 to prevent over-parameterization; floor division ensures proportionality to category count
                </div>
              </div>
            </div>
          </div>

          {/* Config + Why */}
          <div className="fade-up opacity-0 stagger-3">
            <div className="glass-card p-1">
              <div className="px-4 py-3 border-b border-border">
                <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                  <svg className="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Hyperparameters & Justification
                </h3>
              </div>
              <div className="divide-y divide-border/50">
                {config.map((c) => (
                  <div key={c.param} className="px-4 py-2.5 hover:bg-surface-lighter/50 transition-colors">
                    <div className="flex items-center justify-between mb-1">
                      <code className="text-xs font-mono text-text-secondary">{c.param}</code>
                      <span className="text-sm font-mono text-accent font-semibold">{c.value}</span>
                    </div>
                    <p className="text-[10px] text-text-muted leading-relaxed">{c.why}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
