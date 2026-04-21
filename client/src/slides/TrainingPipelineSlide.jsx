export default function TrainingPipelineSlide() {
  const steps = [
    {
      num: '01',
      title: 'Data Loading & Noise Removal',
      detail: 'Load train/test parquet → drop feature_b through feature_g (confirmed noise from Phase 1 EDA) → drop zero-weight rows',
      why: 'Zero-weight rows contribute nothing to the loss metric. Keeping them wastes compute and adds gradient noise. Noise features were proven uncorrelated in Phase 1.',
      color: 'primary',
    },
    {
      num: '02',
      title: 'Label Encoding (Categoricals)',
      detail: 'Sort unique values → assign integer indices → reserve index N for unseen test categories',
      why: 'Sorted encoding ensures determinism across runs. The +1 unseen index prevents crashes on test entities not in train — common in expanding-universe financial data.',
      color: 'accent',
    },
    {
      num: '03',
      title: 'Z-Score Normalization (Numericals)',
      detail: 'Per-feature mean/std from TRAIN only → apply to both train and test → fill residual NaN with 0',
      why: 'Neural networks are sensitive to feature scale (unlike trees). Z-scoring ensures all features contribute equally to gradients. Using train-only stats prevents data leakage from test distribution.',
      color: 'primary',
    },
    {
      num: '04',
      title: 'PyTorch Dataset & DataLoader',
      detail: 'Categorical tensor (int64) + numerical tensor (float32) → batched with shuffle, pin_memory for GPU',
      why: 'Separating cat/num tensors allows the forward pass to route them through different pathways (embedding vs. direct). pin_memory enables async CPU→GPU transfer.',
      color: 'accent',
    },
    {
      num: '05',
      title: 'Training Loop',
      detail: 'MSE loss → AdamW optimizer → cosine schedule with linear warmup → gradient clipping (1.0)',
      why: 'MSE aligns with the competition\'s weighted RMSE metric. Cosine annealing smoothly reduces LR avoiding sudden drops. Warmup prevents early large gradients from destroying initialized embeddings.',
      color: 'primary',
    },
    {
      num: '06',
      title: 'Best Model Selection & Inference',
      detail: 'Save checkpoint at lowest training loss → load best → predict on test in eval mode',
      why: 'Without a validation split, we use training loss as proxy — acceptable because we regularize with dropout/weight-decay. eval() mode disables dropout for deterministic inference.',
      color: 'success',
    },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">Phase 2 / Training Pipeline</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">End-to-End Pipeline</span>
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {steps.map((step, i) => (
            <div key={step.num} className={`glass-card p-4 fade-up opacity-0 stagger-${Math.min(i + 2, 6)} hover:border-${step.color}/30 transition-all`}>
              <div className="flex items-start gap-3">
                <div className={`w-8 h-8 rounded-lg bg-${step.color}/10 flex items-center justify-center shrink-0`}>
                  <span className={`text-xs font-mono font-bold text-${step.color}`}>{step.num}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-semibold text-text-primary mb-1">{step.title}</h4>
                  <p className="text-[11px] text-text-secondary mb-2 leading-relaxed">{step.detail}</p>
                  <div className="bg-surface rounded-lg p-2">
                    <div className="flex items-start gap-1.5">
                      <span className="text-warning text-[10px] font-mono mt-0.5 shrink-0">WHY:</span>
                      <p className="text-[10px] text-text-muted leading-relaxed">{step.why}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
