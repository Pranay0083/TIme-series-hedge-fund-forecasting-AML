export default function PreprocessSlide() {
  const steps = [
    {
      step: '01',
      title: 'Feature Dropping',
      desc: 'Remove low-variance and redundant features identified during EDA',
      color: 'text-danger',
      bg: 'bg-danger/10',
    },
    {
      step: '02',
      title: 'Missing Flags',
      desc: 'Create binary indicators for high-missing features — missingness is a signal',
      color: 'text-warning',
      bg: 'bg-warning/10',
    },
    {
      step: '03',
      title: 'Hierarchical Imputation',
      desc: 'Group-level median fill (code × sub_category) → global median fallback',
      color: 'text-primary-light',
      bg: 'bg-primary/10',
    },
    {
      step: '04',
      title: 'Categorical Encoding',
      desc: 'Label encoding for code, sub_code, sub_category with unknown handling (-1)',
      color: 'text-accent',
      bg: 'bg-accent/10',
    },
    {
      step: '05',
      title: 'Target Processing',
      desc: 'Clip extreme values → horizon-specific z-score normalization (mean/std per horizon)',
      color: 'text-success',
      bg: 'bg-success/10',
    },
    {
      step: '06',
      title: 'Weight Filtering',
      desc: 'Drop zero-weight samples that do not contribute to the competition metric',
      color: 'text-primary-light',
      bg: 'bg-primary/10',
    },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">05 / Preprocessing</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Preprocessing Pipeline</span>
        </h2>

        <div className="relative">
          {/* Vertical line connecting steps */}
          <div className="absolute left-[23px] top-6 bottom-6 w-px bg-gradient-to-b from-primary/50 via-accent/50 to-success/50 hidden md:block" />

          <div className="space-y-4">
            {steps.map((s, i) => (
              <div key={s.step} className={`fade-up opacity-0 stagger-${i + 1} flex items-start gap-5`}>
                <div className={`shrink-0 w-12 h-12 rounded-xl ${s.bg} flex items-center justify-center font-mono text-sm font-bold ${s.color} relative z-10`}>
                  {s.step}
                </div>
                <div className="glass-card p-4 flex-1 hover:border-primary/20 transition-all">
                  <h3 className="text-base font-semibold text-text-primary mb-1">{s.title}</h3>
                  <p className="text-sm text-text-secondary">{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-4 fade-up opacity-0 stagger-6">
          <div className="glass-card p-4 text-center">
            <div className="text-xs text-text-muted uppercase tracking-wider mb-1">Imputation Strategy</div>
            <div className="text-sm text-text-primary font-semibold">Group Median → Global Median</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-xs text-text-muted uppercase tracking-wider mb-1">Pipeline Class</div>
            <code className="text-sm text-accent">PreprocessPipeline</code>
          </div>
        </div>
      </div>
    </div>
  )
}
