import pandas as pd
import corner
import sys
import matplotlib.pyplot as plt

def plot_chain_corner(filename):
    # Load the chain CSV into a DataFrame
    df = pd.read_csv(filename)

    # Exclude metadata columns that are not parameters
    exclude_columns = {"step", "chisq"}
    param_columns = [col for col in df.columns if col not in exclude_columns]

    # Extract parameter data
    param_data = df[param_columns]

    # Create corner plot with dots only, no contours
    fig = corner.corner(
        param_data,
        labels=param_columns,
        show_titles=True,
        plot_contours=False,     # Disable contour plots
        fill_contours=False,     # Also ensure filled contours are off
        plot_density=False       # Disable density shading
    )
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_corner.py path/to/chain.csv")
    else:
        plot_chain_corner(sys.argv[1])
