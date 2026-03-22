import eda1_1 from '../assets/eda1_1.png';
import eda1_2 from '../assets/eda1_2.png';
import eda2_1 from '../assets/eda2_1.png';
import eda2_3 from '../assets/eda2_3.png';
import eda3_1 from '../assets/eda3_1.png';
import eda3_2 from '../assets/eda3_2.png';

const EDALayout = ({ title, subtitle, icon, detail, what, why, image }) => (
  <div className="flex items-center justify-center h-full px-8 md:px-12 py-10 w-full overflow-hidden">
    <div className="max-w-[90rem] w-full flex flex-col h-full">
      <div className="fade-up opacity-0 stagger-1 mb-4 flex-shrink-0">
        <span className="text-sm font-mono text-accent tracking-wider uppercase mb-2 block">{subtitle}</span>
        <div className="flex items-center gap-4">
          <span className="text-4xl bg-surface-lighter p-2 rounded-xl border border-white/5">{icon}</span>
          <h2 className="text-3xl md:text-5xl font-bold">
            <span className="gradient-text">{title}</span>
          </h2>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6 flex-grow min-h-0 h-[calc(100%-8rem)]">
        {/* Large Image Container */}
        <div className="lg:w-[65%] w-full h-[45vh] lg:h-full glass-card flex flex-col items-center justify-center p-6 fade-up opacity-0 stagger-2 bg-[#0B0F19] relative group">
           <img 
             src={image} 
             alt={title} 
             className="w-full h-full object-contain drop-shadow-2xl rounded group-hover:scale-[1.01] transition-transform duration-500" 
           />
        </div>

        {/* Text Analytics Column */}
        <div className="lg:w-[35%] w-full flex flex-col justify-center gap-5 fade-up opacity-0 stagger-3 overflow-y-auto hide-scrollbar pb-2">
           <div className="glass-card p-6 border-l-[6px] border-l-primary shadow-lg hover:border-primary-light transition-colors bg-surface-darker/60">
              <h4 className="text-xs font-bold text-text-primary uppercase tracking-widest mb-3 opacity-80">Key Finding</h4>
              <p className="text-lg font-medium text-text-primary leading-relaxed">{detail}</p>
           </div>

           <div className="glass-card p-6 border-l-[6px] border-l-accent shadow-lg bg-surface-darker/60">
              <h4 className="text-xs font-bold text-accent uppercase tracking-widest mb-3">What & How?</h4>
              <p className="text-[15px] text-text-secondary leading-relaxed">{what}</p>
           </div>

           <div className="glass-card p-6 border-l-[6px] border-l-success shadow-lg bg-surface-darker/60">
              <h4 className="text-xs font-bold text-success uppercase tracking-widest mb-3">Why do we care?</h4>
              <p className="text-[15px] text-text-secondary leading-relaxed">{why}</p>
           </div>
        </div>
      </div>
    </div>
  </div>
);

export function EDAVariance() {
  return <EDALayout 
    subtitle="04.1 / Target Analysis"
    icon="📈"
    title="Structural Target Variance"
    detail="Target variance strictly scales with the forecasting horizon. Standard deviation grows enormously from horizon 1 to 25."
    what="We aggregated target standard deviation broken down by horizon. The graph visualizes the rapid exponential scaling of variance as the forecasting window increases."
    why="Identified the core necessity to normalize the target scaling per horizon. Without this, long-horizon variances completely dominate the loss function, wiping out short-term signals."
    image={eda1_1} 
  />
}

export function EDAKurtosis() {
  return <EDALayout 
    subtitle="04.2 / Target Analysis"
    icon="📊"
    title="Extreme Kurtosis & Fat Tails"
    detail="The target variable (y_target) features an extreme kurtosis (~290) tightly wrapped around a near-zero center with highly fat tails."
    what="Analyzed the target distribution using deep histograms and QQ-plots. Even after transformations, the data massively diverges from a standard Gaussian."
    why="Established that standard loss functions (like MSE) will fail completely due to extreme sensitivity to fat tail outliers. Robust losses like Huber or MAE are mandatory to prevent gradient explosion."
    image={eda1_2} 
  />
}

export function EDALeakage() {
  return <EDALayout 
    subtitle="04.3 / Integrity & Leakage"
    icon="🛑"
    title="Unseen Test Entities"
    detail="Train and test sets share only 12 sub_code entities; 35 test entities are completely unseen in the training historical data."
    what="Performed strict categorical overlap analysis between the train and test data sets to verify distributional equivalence."
    why="Proved that naive target-encoding by the finite sub_code sets causes massive leakage and catastrophic failure on the test set. Models are forced to generalize using macro 'code' groupings instead."
    image={eda2_1} 
  />
}

export function EDAWeights() {
  return <EDALayout 
    subtitle="04.4 / Integrity & Leakage"
    icon="⚖️"
    title="Extreme Weight Discrepancies"
    detail="Samples contain massive scoring-weight discrepancies spanning 13 orders of magnitude, tightly bundled around the y=0 mark."
    what="Examined the competition's sample weights plotted natively on a logarithmic scale against target distributions."
    why="If these heavy, extreme weights are used naively in a loss function, they induce severe gradient instability during model updates. Requires intelligent log-transformation or rank-based scaling during early training."
    image={eda2_3} 
  />
}

export function EDAUniverse() {
  return <EDALayout 
    subtitle="04.5 / Temporal Shifts"
    icon="🌌"
    title="Expanding Universe Growth"
    detail="The cross-sectional universe size grows systematically over time. Early indexes are highly sparse compared to saturated recent data."
    what="Plotted live universe entity counts sequentially traversing across the discrete ts_index time axis to measure chronological density."
    why="Models trained equally on early sparse data might fail to capture the complex cross-sectional interactions natively present in the dense, modern regime over-representing recent index density."
    image={eda3_1} 
  />
}

export function EDACovariance() {
  return <EDALayout 
    subtitle="04.6 / Temporal Shifts"
    icon="🌪️"
    title="Unstable Covariance Structures"
    detail="The underlying covariance structure fundamentally destabilizes and shifts phase across different temporal market regimes."
    what="Tracked the Frobenius norm of feature correlation matrices across distinct temporal rolling windows to isolate systemic drift."
    why="Exposed the sheer danger of standard randomized k-fold splits. Proved the explicit need for moving to an expanding-window chronological validation strategy to avoid lookahead biases."
    image={eda3_2} 
  />
}
