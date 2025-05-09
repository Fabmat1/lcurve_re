import pandas as pd
import corner
import sys
import matplotlib.pyplot as plt

def plot_chain_corner(filename):
    # Load the chain CSV into a DataFrame
    df = pd.read_csv(filename)

    # Exclude metadata columns that are not parameters
    exclude_columns = {"step", "chisq"}
    param_data = df.loc[:, ~df.columns.isin(exclude_columns)]

    # Extract labels from actual DataFrame to preserve correct order
    labels = param_data.columns.tolist()

    # Create corner plot with only scatter points
    fig = corner.corner(
        param_data.values,  # Pass raw values to avoid label mismatch
        labels=labels,
        show_titles=True,
        plot_contours=True,
        fill_contours=False,
        plot_density=True
    )
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_corner.py path/to/chain.csv")
    else:
        plot_chain_corner(sys.argv[1])
