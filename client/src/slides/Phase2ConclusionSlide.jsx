export default function Phase2ConclusionSlide() {
  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">Phase 2 / Summary</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Phase 2 Summary & Next Steps</span>
        </h2>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* What we achieved */}
          <div className="space-y-4 fade-up opacity-0 stagger-2">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
              <div className="w-6 h-0.5 bg-success" />
              Phase 2 Achievements
            </h3>
            {[
              'Built MLP + Entity Embeddings pipeline from scratch in PyTorch — full end-to-end differentiable system',
              'Learned dense categorical representations replacing 500+ sparse one-hot columns with ≤50-dim embeddings per feature',
              'Implemented production-grade training: AdamW, cosine warmup scheduler, gradient clipping, and proper weight initialization',
              'Handled expanding-universe problem natively with unseen-category embedding indices',
              'Established deep learning baseline with systematic progression: LightGBM → MLP+Embeddings → FT-Transformer',
              'Built reproducibility infrastructure: seed fixes for NumPy, PyTorch, CUDA across all random sources',
            ].map((item, i) => (
              <div key={i} className="flex items-start gap-3">
                <div className="w-5 h-5 rounded-full bg-success/10 flex items-center justify-center shrink-0 mt-0.5">
                  <svg className="w-3 h-3 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-sm text-text-secondary leading-relaxed">{item}</p>
              </div>
            ))}
          </div>

          {/* Next steps */}
          <div className="space-y-4 fade-up opacity-0 stagger-3">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
              <div className="w-6 h-0.5 bg-accent" />
              Next Steps
            </h3>
            {[
              {
                title: 'FT-Transformer (Feature Tokenizer + Transformer)',
                desc: 'Already implemented. Each feature becomes a token — multi-head self-attention captures cross-feature interactions that MLP\'s concatenation cannot.',
                why: 'MLP treats all feature interactions equally via concatenation. Attention selectively weights which features matter for each prediction.',
              },
              {
                title: 'Ensemble: Trees + Neural Networks',
                desc: 'Blend LightGBM predictions with MLP/Transformer predictions. Different model families capture different patterns.',
                why: 'Trees excel at sharp threshold-based splits; neural nets excel at smooth non-linear manifolds. Ensembling captures both.',
              },
              {
                title: 'Weighted MSE Training Loss',
                desc: 'Incorporate competition sample weights directly into the loss function instead of uniform MSE.',
                why: 'Current MSE treats all samples equally. The competition metric weights them — aligning training loss with evaluation metric improves final score directly.',
              },
              {
                title: 'Per-Horizon Neural Networks',
                desc: 'Train separate MLP models for each horizon, with horizon-specific hyperparameters.',
                why: 'Phase 1 showed signal-to-noise ratio varies drastically by horizon. A shared model is a compromise — dedicated models can specialize.',
              },
            ].map((item) => (
              <div key={item.title} className="glass-card p-4 hover:border-accent/30 transition-all">
                <h4 className="text-sm font-semibold text-text-primary mb-1">{item.title}</h4>
                <p className="text-xs text-text-secondary mb-2">{item.desc}</p>
                <div className="bg-surface rounded-lg p-2">
                  <div className="flex items-start gap-1.5">
                    <span className="text-warning text-[10px] font-mono mt-0.5 shrink-0">WHY:</span>
                    <p className="text-[10px] text-text-muted leading-relaxed">{item.why}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Updated tech stack */}
        <div className="mt-6 glass-card p-5 fade-up opacity-0 stagger-4">
          <h3 className="text-xs text-text-muted uppercase tracking-wider mb-3">Phase 2 Technology Stack</h3>
          <div className="flex flex-wrap gap-2">
            {['Python', 'PyTorch', 'LightGBM', 'Entity Embeddings', 'AdamW', 'Cosine Annealing', 'BatchNorm', 'Pandas', 'NumPy', 'Parquet'].map((tech) => (
              <span key={tech} className="px-3 py-1.5 rounded-lg bg-surface-lighter border border-border text-xs font-mono text-text-secondary">
                {tech}
              </span>
            ))}
          </div>
        </div>

        {/* Sign off */}
        <div className="mt-4 text-center fade-up opacity-0 stagger-5">
          <p className="text-lg text-text-muted">
            Thank you — <span className="gradient-text font-semibold">Team Bias and Variance</span>
          </p>
        </div>
      </div>
    </div>
  )
}
