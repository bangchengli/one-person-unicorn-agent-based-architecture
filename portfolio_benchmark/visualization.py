import matplotlib.pyplot as plt
from .config import CHART_SAVE_DIR

def visualize_cost_efficiency(df,
                              x_col: str = "est_api_cost_usd",
                              y_col: str = "sharpe",
                              label_col: str = "model",
                              title: str = "Cost Performance Frontier",
                              outfile: str = None):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print("visualize_cost_efficiency: missing columns, skip plot")
        return
    plt.figure()
    xs = df[x_col].values
    ys = df[y_col].values
    plt.scatter(xs, ys)
    labels = df[label_col].astype(str).tolist()
    for i, txt in enumerate(labels):
        plt.annotate(txt, (xs[i], ys[i]))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    if outfile is None:
        outfile = CHART_SAVE_DIR / "cost_performance_frontier.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {outfile}")
