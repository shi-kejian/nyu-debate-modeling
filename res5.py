import matplotlib.pyplot as plt
import numpy as np

# --- Data Structure (PLACEHOLDERS - FILL THESE IN) ---
results = {
    'L1': {
        'Unassisted': 38.5,
        'SC': 47.5,
        'DC': 52.8,
        'Debate': 54.5,
    },
    'L2': {
        'Unassisted': 40.2,
        'SC': 48.2,
        'DC': 52.0,
        'Debate': 54.0,
    },
    'L3': {
        'Unassisted': 45.8,
        'SC': 55.1,
        'DC': 53.5,
        'Debate': 52.5,
    },
}

# --- Plotting Function (TRANSPOSED) ---

def plot_protocol_comparison(results):
    """
    Generates a bar chart comparing the performance of different oversight
    protocols (SC, DC, Debate) and the unassisted baseline across different
    judge expertise levels (L1, L2, L3).  X-axis is now Judge Level.
    """

    judge_levels = list(results.keys())  # ['L1', 'L2', 'L3']
    protocols = ['Unassisted', 'SC', 'DC', 'Debate']
    bar_width = 0.2
    index = np.arange(len(judge_levels))  # Now we have 3 groups (L1, L2, L3)

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['gray', 'skyblue', 'lightgreen', 'coral']  # Colors for protocols

    for i, protocol in enumerate(protocols):
        values = [results[level][protocol] for level in judge_levels]
        ax.bar(index + i * bar_width, values, bar_width,
               label=f'{protocol}', color=colors[i])

    ax.set_xlabel('Judge Expertise Level')  # Clearer x-axis label
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Comparison of Oversight Protocols Across Judge Expertise Levels')
    ax.set_xticks(index + 1.5 * bar_width)  # Center x-ticks
    ax.set_xticklabels(judge_levels)  # L1, L2, L3 on x-axis
    ax.legend(title="Oversight Protocol") # add title to the legend.
    ax.set_ylim([0, 100])
    plt.tight_layout()
    plt.savefig("protocol_comparison_transposed.png")
    plt.show()

# --- Generate the Plot ---
plot_protocol_comparison(results)