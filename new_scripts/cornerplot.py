import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.gridspec import GridSpec

def weighted_quantile(values, quantiles, weights):
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    cumw = np.cumsum(w)
    cumw /= cumw[-1]
    return np.interp(quantiles, cumw, v)

def format_with_error(med, lo, hi):
    err = max(hi - med, med - lo)
    if err == 0 or np.isnan(err):
        return f"{med:.2f}", f"+{(hi-med):.2f}/-{(med-lo):.2f}"
    sig = 2
    decimals = max(-int(np.floor(np.log10(err))) + (sig - 1), 0)
    fmt = f"{{:.{decimals}f}}"
    med_str = fmt.format(med)
    err_hi = fmt.format(hi - med)
    err_lo = fmt.format(med - lo)
    return med_str, f"+{err_hi}/-{err_lo}"

def sync_y_to_x(ax_src, ax_dst):
    def on_ylim_change(event_ax):
        if getattr(ax_dst, '_updating', False):
            return
        ax_dst._updating = True
        try:
            ylim = ax_src.get_ylim()
            ax_dst.set_xlim(ylim)
            ax_dst.figure.canvas.draw_idle()
        finally:
            ax_dst._updating = False
    return on_ylim_change

def sync_x_to_y(ax_src, ax_dst):
    def on_xlim_change(event_ax):
        if getattr(ax_dst, '_updating', False):
            return
        ax_dst._updating = True
        try:
            xlim = ax_src.get_xlim()
            ax_dst.set_ylim(xlim)
            ax_dst.figure.canvas.draw_idle()
        finally:
            ax_dst._updating = False
    return on_xlim_change

def get_quantile_bounds(values, weights, quantile=0.995, margin=0.05):
    """Get bounds covering the specified quantile of data with margin."""
    q_low = (1 - quantile) / 2
    q_high = 1 - q_low
    low_val, high_val = weighted_quantile(values, [q_low, q_high], weights)
    range_size = high_val - low_val
    margin_size = range_size * margin
    return low_val - margin_size, high_val + margin_size

def cornerplot_from_chain(chain_file, burnin_frac=0.1, thin=1, n_bins=75):
    """
    Create a corner plot from a chain file.
    
    Parameters:
    -----------
    chain_file : str
        Path to the chain file
    burnin_frac : float
        Fraction of chain to discard as burn-in (default: 0.1)
    thin : int
        Thinning factor for the chain (default: 1)
    """
    
    # Load the chain file
    df = pd.read_csv(chain_file)
    
    # Get parameter names (exclude 'step' and 'chisq')
    param_names = [col for col in df.columns if col not in ['step', 'chisq']]
    n_params = len(param_names)
    
    # Apply burn-in and thinning
    n_samples = len(df)
    burnin = int(burnin_frac * n_samples)
    df = df.iloc[burnin::thin]
    
    print(f"Using {len(df)} samples after burn-in ({burnin} samples) and thinning (factor {thin})")
    
    # Extract parameter data
    data = {}
    for param in param_names:
        data[param] = df[param].values
    
    # Compute 2D histograms
    H_2d = {}
    edges = {}
    
    # First pass: determine edges
    for param in param_names:
        # Use percentiles for robust edge determination
        vmin, vmax = np.percentile(data[param], [0.5, 99.5])
        margin = 0.1 * (vmax - vmin)
        edges[param] = np.linspace(vmin - margin, vmax + margin, n_bins + 1)
    
    # Compute all 2D histograms
    for i, param1 in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            if i > j:  # Only compute lower triangle
                hist, _, _ = np.histogram2d(
                    data[param2], data[param1],
                    bins=[edges[param2], edges[param1]]
                )
                H_2d[(param1, param2)] = hist.T  # Transpose to match expected format
    
    # Compute 1D marginals
    marg = {}
    for param in param_names:
        hist, _ = np.histogram(data[param], bins=edges[param])
        marg[param] = hist.astype(float)
    
    # Compute bounds for zooming
    bounds = {}
    for param in param_names:
        centers = 0.5 * (edges[param][1:] + edges[param][:-1])
        wts = marg[param] / marg[param].sum()
        lo, hi = get_quantile_bounds(centers, wts, quantile=0.995, margin=0.05)
        bounds[param] = (lo, hi)
    
    # Print parameter estimates
    print("\nParameter estimates (median +/- 68% CI):")
    print("-" * 50)
    for param in param_names:
        centers = 0.5 * (edges[param][1:] + edges[param][:-1])
        wts = marg[param] / marg[param].sum()
        med, lo, hi = weighted_quantile(centers, [0.5, 0.16, 0.84], wts)
        s, err = format_with_error(med, lo, hi)
        print(f"{param:>15s} = {s:>10s} ({err})")
    
    # Create corner plot
    n = len(param_names)
    fig = plt.figure(figsize=(2.5 * n, 2.5 * n))
    gs = GridSpec(n, n, left=0.1, right=0.95, bottom=0.1, top=0.95,
                  wspace=0.05, hspace=0.05)
    
    axes = {}
    
    # Create all axes
    for i, param1 in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            ax = fig.add_subplot(gs[i, j])
            axes[(i, j)] = ax
    
    # Fill in the plots
    for i, param1 in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            ax = axes[(i, j)]
            
            if i > j:
                # Lower triangle: 2D histograms
                arr = H_2d[(param1, param2)]
                
                # Mask zeros
                mask = (arr <= 0) | np.isnan(arr)
                arrm = np.ma.masked_array(arr, mask=mask)
                
                if arrm.count() > 0:
                    vmin = float(arrm.compressed().min())
                    vmax = float(arrm.compressed().max())
                else:
                    vmin, vmax = 1e-2, 1.0
                
                ax.pcolormesh(
                    edges[param2], edges[param1], arrm,
                    cmap='Blues',
                    vmin=vmin,
                    vmax=vmax,
                    shading='auto'
                )
                
                # Handle tick labels
                if i < n - 1:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.set_xlabel(param2)
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                    for lbl in ax.get_xticklabels():
                        lbl.set_rotation(45)
                
                if j > 0:
                    ax.tick_params(axis='y', labelleft=False)
                else:
                    ax.set_ylabel(param1)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                    
            elif i == j:
                # Diagonal: 1D histograms
                centers = 0.5 * (edges[param1][1:] + edges[param1][:-1])
                ax.hist(
                    centers, bins=edges[param1], weights=marg[param1],
                    histtype='stepfilled',
                    edgecolor='black', facecolor='lightblue', alpha=0.6
                )
                
                # Handle tick labels
                if i < n - 1:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.set_xlabel(param1)
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                    for lbl in ax.get_xticklabels():
                        lbl.set_rotation(45)
                
                ax.tick_params(axis='y', labelleft=False)
                
            else:
                # Upper triangle: turn off
                ax.axis('off')
    
    # Set up axis synchronization
    # For each parameter, synchronize all its representations
    for k, param in enumerate(param_names):
        # Get all axes where this parameter appears on x-axis
        x_axes = [axes[(i, k)] for i in range(k, n)]
        
        # Get all axes where this parameter appears on y-axis
        y_axes = [axes[(k, j)] for j in range(k)]
        
        # Share x-axes among themselves
        if len(x_axes) > 1:
            master_x = x_axes[0]
            for ax in x_axes[1:]:
                ax.sharex(master_x)
        
        # Share y-axes among themselves
        if len(y_axes) > 1:
            master_y = y_axes[0]
            for ax in y_axes[1:]:
                ax.sharey(master_y)
        
        # Connect x and y representations bidirectionally
        for ax_x in x_axes:
            for ax_y in y_axes:
                ax_x.callbacks.connect("xlim_changed", sync_x_to_y(ax_x, ax_y))
                ax_y.callbacks.connect("ylim_changed", sync_y_to_x(ax_y, ax_x))
    
    # Apply initial zoom to 99.5% bounds
    for i, param in enumerate(param_names):
        # Set bounds on diagonal plots (which will propagate through sharing)
        diag_ax = axes[(i, i)]
        diag_ax.set_xlim(bounds[param])
    
    plt.suptitle(f"Corner plot: {os.path.basename(chain_file)}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create corner plot from MCMC chain file")
    parser.add_argument("chain_file", type=str, help="Path to the chain file")
    parser.add_argument("--burnin", type=float, default=0.1,
                        help="Fraction of chain to discard as burn-in (default: 0.1)")
    parser.add_argument("--thin", type=int, default=1,
                        help="Thinning factor for the chain (default: 1)")
    parser.add_argument("--nbins", type=int, default=75,
                        help="Number of histogram bins. (default: 75)")
    args = parser.parse_args()
    
    cornerplot_from_chain(args.chain_file, burnin_frac=args.burnin, thin=args.thin, n_bins=args.nbins)