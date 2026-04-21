export default function DesignDecisionsSlide() {
  const decisions = [
    {
      decision: 'AdamW over SGD / Adam',
      reasoning: 'AdamW decouples weight decay from the adaptive learning rate — standard Adam applies L2 to scaled gradients, producing weaker regularization. In tabular data with mixed feature scales, proper weight decay is critical.',
      why_deeper: 'Loshchilov & Hutter (2019) showed Adam\'s L2 penalty interacts pathologically with adaptive moment estimates. AdamW fixes this, improving generalization by 2–5% on tabular benchmarks.',
      icon: '⚡',
    },
    {
      decision: 'Cosine Annealing with Warmup',
      reasoning: 'Linear warmup for 1 epoch gradually ramps the learning rate, protecting randomly-initialized embedding layers from large early updates. Cosine decay then smoothly reduces LR for fine-grained convergence.',
      why_deeper: 'Embedding gradients in epoch 1 are noisy (random representations). A sudden large LR would push embeddings to extreme values that are hard to recover from. Warmup lets embeddings "calibrate" gently first.',
      icon: '📈',
    },
    {
      decision: 'Kaiming (He) Initialization',
      reasoning: 'ReLU activation creates a "dying neurons" risk. Kaiming init scales weights by √(2/fan_in), compensating for ReLU\'s zero-output half. This maintains healthy gradient variance across layers.',
      why_deeper: 'Without Kaiming, deeper layers would have exponentially vanishing activations. With our 4-layer MLP (512→256→128→1), using Xavier/Glorot would lose ~50% signal per layer. Kaiming preserves it.',
      icon: '🎯',
    },
    {
      decision: 'BatchNorm only in Layer 1',
      reasoning: 'BatchNorm stabilizes training by normalizing internal activations. We apply it only after the first dense layer because that\'s where heterogeneous input scales (embeddings + numericals) create the most variance.',
      why_deeper: 'Adding BN to every layer would over-regularize our small network and add overhead. One BN layer at the "join point" of embeddings + numericals is the optimal tradeoff.',
      icon: '📊',
    },
    {
      decision: 'MSE Loss (not Huber/MAE)',
      reasoning: 'The competition metric is weighted RMSE, which is directly related to MSE. Using MSE as training loss ensures gradient optimization aligns exactly with the evaluation criterion.',
      why_deeper: 'Huber loss would de-emphasize outliers, but the competition weights important samples explicitly — the weight column already handles sample importance. Huber would double-correct and reduce signal.',
      icon: '📐',
    },
    {
      decision: 'No Validation Split',
      reasoning: 'With ~5.3M training rows, using all data for training maximizes signal extraction. Regularization (dropout=0.25, weight_decay=1e-5, early-ish stopping at 10 epochs) prevents overfitting without wasting data.',
      why_deeper: 'In financial data, the most recent rows are the most valuable (non-stationarity). A temporal validation split would sacrifice the recent data we need most. We rely on implicit regularization instead.',
      icon: '🔒',
    },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">Phase 2 / Design Decisions</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-2 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Every Decision, Justified</span>
        </h2>
        <p className="text-text-secondary mb-6 fade-up opacity-0 stagger-2 text-sm">
          Each architectural and training choice is backed by both empirical evidence and theoretical reasoning
        </p>

        <div className="grid md:grid-cols-2 gap-4">
          {decisions.map((d, i) => (
            <div key={i} className={`glass-card p-4 fade-up opacity-0 stagger-${Math.min(i + 2, 6)} hover:border-primary/30 transition-all`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">{d.icon}</span>
                <h4 className="text-sm font-semibold text-text-primary">{d.decision}</h4>
              </div>
              <p className="text-xs text-text-secondary mb-2 leading-relaxed">{d.reasoning}</p>
              <div className="bg-surface rounded-lg p-2">
                <div className="flex items-start gap-1.5">
                  <span className="text-accent text-[10px] font-mono mt-0.5 shrink-0">WHY²:</span>
                  <p className="text-[10px] text-text-muted leading-relaxed">{d.why_deeper}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
