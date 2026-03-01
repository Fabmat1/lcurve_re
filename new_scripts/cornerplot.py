import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.gridspec import GridSpec

# Default LaTeX labels for common parameters
DEFAULT_PARAM_LABELS = {
    'q': r'$q$',
    'velocity_scale': r'$v_s$ [$\mathrm{km}\, \mathrm{s}^{-1}$]',
    'iangle': r'$i\,[\mathrm{deg}]$',
    'r1': r'$r_{\mathrm{sd}}$',
}

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

_sync_updating = False

def sync_y_to_x(ax_src, ax_dst):
    def on_ylim_change(event_ax):
        global _sync_updating
        if _sync_updating:
            return
        _sync_updating = True
        try:
            ylim = ax_src.get_ylim()
            ax_dst.set_xlim(ylim)
            ax_dst.figure.canvas.draw_idle()
        finally:
            _sync_updating = False
    return on_ylim_change

def sync_x_to_y(ax_src, ax_dst):
    def on_xlim_change(event_ax):
        global _sync_updating
        if _sync_updating:
            return
        _sync_updating = True
        try:
            xlim = ax_src.get_xlim()
            ax_dst.set_ylim(xlim)
            ax_dst.figure.canvas.draw_idle()
        finally:
            _sync_updating = False
    return on_xlim_change

def get_quantile_bounds(values, weights, quantile=0.995, margin=0.05):
    """Get bounds covering the specified quantile of data with margin."""
    q_low = (1 - quantile) / 2
    q_high = 1 - q_low
    low_val, high_val = weighted_quantile(values, [q_low, q_high], weights)
    range_size = high_val - low_val
    margin_size = range_size * margin
    return low_val - margin_size, high_val + margin_size

def parse_param_limits(limits_str):
    """
    Parse parameter limits string into a dictionary.
    Format: 'param1:min,max;param2:min,max'
    Example: 'mass:0.5,2.0;radius:1.0,3.0'
    """
    if not limits_str:
        return {}
    
    limits = {}
    for item in limits_str.split(';'):
        item = item.strip()
        if not item:
            continue
        param, bounds = item.split(':')
        min_val, max_val = bounds.split(',')
        limits[param.strip()] = (float(min_val), float(max_val))
    return limits

def parse_param_labels(labels_str):
    """
    Parse parameter labels string into a dictionary.
    Format: 'param1:label1;param2:label2'
    Example: 'mass:$M$ [$M_\\odot$];radius:$R$ [$R_\\odot$]'
    """
    if not labels_str:
        return {}
    
    labels = {}
    for item in labels_str.split(';'):
        item = item.strip()
        if not item:
            continue
        # Split on first colon only (label may contain colons)
        idx = item.index(':')
        param = item[:idx].strip()
        label = item[idx+1:].strip()
        labels[param] = label
    return labels

def get_param_label(param, custom_labels=None):
    """
    Get the display label for a parameter.
    Priority: custom_labels > DEFAULT_PARAM_LABELS > param name
    """
    if custom_labels and param in custom_labels:
        return custom_labels[param]
    if param in DEFAULT_PARAM_LABELS:
        return DEFAULT_PARAM_LABELS[param]
    return param

def cornerplot_from_chain(chain_file, burnin_frac=0.1, thin=1, n_bins=75,
                          figsize=None, fontsize=10, ticksize=None,
                          output=None, dpi=150, param_limits=None,
                          linewidth=1.0, pad_inches=0.0, axiswidth=1.0,
                          inner_ticks=False, param_labels=None, usetex=False):
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
    n_bins : int
        Number of histogram bins (default: 75)
    figsize : tuple of float
        Figure size (width, height) in inches. Default scales with n_params.
    fontsize : float
        Font size for axis labels (default: 10)
    ticksize : float
        Font size for tick labels. Defaults to fontsize - 2 if not specified.
    output : str
        Path to save the output PDF. If None, displays interactively.
    dpi : int
        DPI for the saved figure (default: 150)
    param_limits : dict
        Dictionary of parameter limits: {'param_name': (min, max), ...}
        If None, limits are determined automatically from the data.
    linewidth : float
        Line width for histogram edges (default: 1.0)
    pad_inches : float
        Padding around figure when saving (default: 0.0 for no padding)
    axiswidth : float
        Line width for axis spines/borders (default: 1.0)
    inner_ticks : bool
        If True, show tick marks on all subplots. If False (default),
        show tick marks only on the outside (bottom row and left column).
    param_labels : dict
        Dictionary of custom parameter labels: {'param_name': 'LaTeX label', ...}
        Overrides default labels. If a parameter is not in this dict or
        DEFAULT_PARAM_LABELS, the column name is used.
    usetex : bool
        If True, use LaTeX for text rendering (requires LaTeX installation).
        If False (default), use matplotlib's mathtext.
    """
    
    # Configure LaTeX rendering
    if usetex:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
    else:
        plt.rcParams['text.usetex'] = False
    
    # Set tick size if not specified
    if ticksize is None:
        ticksize = fontsize - 2
    
    # Initialize param_limits if not provided
    if param_limits is None:
        param_limits = {}
    
    # Initialize param_labels if not provided
    if param_labels is None:
        param_labels = {}
    
    # Load the chain file
    df = pd.read_csv(chain_file)
    
    # Get parameter names (exclude 'step' and 'chisq')
    param_names = [col for col in df.columns if col not in ['step', 'chisq']]
    n_params = len(param_names)
    
    # Set default figsize based on number of parameters
    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)
    
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
    
    # First pass: determine edges (use param_limits if provided)
    for param in param_names:
        if param in param_limits:
            # Use user-specified limits
            vmin, vmax = param_limits[param]
        else:
            # Use percentiles for robust edge determination
            vmin, vmax = np.percentile(data[param], [0.5, 99.5])
            margin = 0.1 * (vmax - vmin)
            vmin -= margin
            vmax += margin
        edges[param] = np.linspace(vmin, vmax, n_bins + 1)
    
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
    
    # Compute bounds for zooming (use param_limits if provided, otherwise auto)
    bounds = {}
    for param in param_names:
        if param in param_limits:
            bounds[param] = param_limits[param]
        else:
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
    
    # Create corner plot with minimal margins
    n = len(param_names)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n, n, figure=fig, left=0.08, right=0.98, bottom=0.08, top=0.98,
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
            
            # Set axis spine width
            for spine in ax.spines.values():
                spine.set_linewidth(axiswidth)
            
            # Set tick label size for all axes
            ax.tick_params(axis='both', labelsize=ticksize, width=axiswidth)
            
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
                
                # Use edgecolor='face', linewidth=0, and rasterized=True to eliminate white lines
                ax.pcolormesh(
                    edges[param2], edges[param1], arrm,
                    cmap='Blues',
                    vmin=vmin,
                    vmax=vmax,
                    shading='auto',
                    edgecolor='face',
                    linewidth=0,
                    rasterized=True
                )
                
                # Handle tick marks and labels
                if i < n - 1:
                    ax.tick_params(axis='x', labelbottom=False)
                    if not inner_ticks:
                        ax.tick_params(axis='x', bottom=False)
                else:
                    ax.set_xlabel(get_param_label(param2, param_labels), fontsize=fontsize)
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                    for lbl in ax.get_xticklabels():
                        lbl.set_rotation(45)
                
                if j > 0:
                    ax.tick_params(axis='y', labelleft=False)
                    if not inner_ticks:
                        ax.tick_params(axis='y', left=False)
                else:
                    ax.set_ylabel(get_param_label(param1, param_labels), fontsize=fontsize)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                    
            elif i == j:
                # Diagonal: 1D histograms
                centers = 0.5 * (edges[param1][1:] + edges[param1][:-1])
                ax.hist(
                    centers, bins=edges[param1], weights=marg[param1],
                    histtype='stepfilled',
                    edgecolor='black', facecolor='lightblue', alpha=0.6,
                    linewidth=linewidth
                )
                
                # Handle tick marks and labels
                if i < n - 1:
                    ax.tick_params(axis='x', labelbottom=False)
                    if not inner_ticks:
                        ax.tick_params(axis='x', bottom=False)
                else:
                    ax.set_xlabel(get_param_label(param1, param_labels), fontsize=fontsize)
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
                    for lbl in ax.get_xticklabels():
                        lbl.set_rotation(45)
                
                # Y-axis on diagonal: no labels, ticks depend on inner_ticks
                ax.tick_params(axis='y', labelleft=False)
                if not inner_ticks:
                    ax.tick_params(axis='y', left=False)
                
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
    
    # Apply bounds (either user-specified or auto-computed)
    for i, param in enumerate(param_names):
        diag_ax = axes[(i, i)]
        diag_ax.set_xlim(bounds[param])
    
    # Save or show the figure
    if output is not None:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches,
                    facecolor='white', edgecolor='none')
        print(f"\nFigure saved to: {output} (dpi={dpi})")
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create corner plot from MCMC chain file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default parameter labels:
  q             -> $q$
  velocity_scale -> $v_s$ [$\\mathrm{km}\\, \\mathrm{s}^{-1}$]
  iangle        -> $i\\,[\\mathrm{deg}]$
  r1            -> $r_{\\mathrm{sd}}$

Examples:
  %(prog)s chain.csv --output corner.pdf
  %(prog)s chain.csv --figsize 10 10 --fontsize 12 --dpi 300 -o corner.pdf
  %(prog)s chain.csv --limits 'mass:0.5,2.0;radius:1.0,3.0' -o corner.pdf
  %(prog)s chain.csv --labels 'mass:$M$ [$M_\\odot$];temp:$T$ [K]' -o corner.pdf
  %(prog)s chain.csv --usetex --fontsize 14 -o corner.pdf
        """
    )
    parser.add_argument("chain_file", type=str, help="Path to the chain file")
    parser.add_argument("--burnin", type=float, default=0.1,
                        help="Fraction of chain to discard as burn-in (default: 0.1)")
    parser.add_argument("--thin", type=int, default=1,
                        help="Thinning factor for the chain (default: 1)")
    parser.add_argument("--nbins", type=int, default=75,
                        help="Number of histogram bins (default: 75)")
    parser.add_argument("--figsize", type=float, nargs=2, default=None,
                        metavar=('WIDTH', 'HEIGHT'),
                        help="Figure size in inches (default: 2.5*n_params for each dimension)")
    parser.add_argument("--fontsize", type=float, default=10,
                        help="Font size for axis labels (default: 10)")
    parser.add_argument("--ticksize", type=float, default=None,
                        help="Font size for tick labels (default: fontsize - 2)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path for saving the figure (e.g., output.pdf)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for the saved figure (default: 150)")
    parser.add_argument("--limits", type=str, default=None,
                        help="Parameter limits in format 'param1:min,max;param2:min,max'")
    parser.add_argument("--linewidth", type=float, default=1.0,
                        help="Line width for histogram edges (default: 1.0)")
    parser.add_argument("--pad", type=float, default=0.0,
                        help="Padding around figure in inches when saving (default: 0.0)")
    parser.add_argument("--axiswidth", type=float, default=1.0,
                        help="Line width for axis spines/borders (default: 1.0)")
    parser.add_argument("--inner-ticks", action="store_true", default=False,
                        help="Show tick marks on all subplots, not just the outside edges")
    parser.add_argument("--labels", type=str, default=None,
                        help="Custom parameter labels in format 'param1:$label1$;param2:$label2$'")
    parser.add_argument("--usetex", action="store_true", default=False,
                        help="Use LaTeX for text rendering (requires LaTeX installation)")
    
    args = parser.parse_args()
    
    # Convert figsize list to tuple if provided
    figsize = tuple(args.figsize) if args.figsize else None
    
    # Parse parameter limits
    param_limits = parse_param_limits(args.limits)
    
    # Parse parameter labels
    param_labels = parse_param_labels(args.labels)
    
    cornerplot_from_chain(
        args.chain_file,
        burnin_frac=args.burnin,
        thin=args.thin,
        n_bins=args.nbins,
        figsize=figsize,
        fontsize=args.fontsize,
        ticksize=args.ticksize,
        output=args.output,
        dpi=args.dpi,
        param_limits=param_limits,
        linewidth=args.linewidth,
        pad_inches=args.pad,
        axiswidth=args.axiswidth,
        inner_ticks=args.inner_ticks,
        param_labels=param_labels,
        usetex=args.usetex
    )