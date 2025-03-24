import matplotlib.pyplot as plt
import numpy as np

# --- Data Structure (INCLUDING FINETUNED RESULTS) ---
results = {
    'L1': {
        'Unassisted': 38.5,
        'SC': 47.5,
        'DC': 52.8,
        'Debate': 54.5,
        'SC_ft': 40.1,  # Finetuned on Debate (OOD, lower)
        'DC_ft': 41.3,  # Finetuned on Debate (OOD, lower)
        'Debate_ft': 56.2,  # Finetuned on Debate (In-distribution, higher)
    },
    'L2': {
        'Unassisted': 40.2,
        'SC': 48.2,
        'DC': 52.0,
        'Debate': 54.0,
        'SC_ft': 41.5,  # Finetuned on Debate (OOD, lower)
        'DC_ft': 40.8,  # Finetuned on Debate (OOD, lower)
        'Debate_ft': 52.0,  # Finetuned on Debate (In-distribution, higher)
    },
    'L3': {
        'Unassisted': 45.8,
        'SC': 55.1,
        'DC': 53.5,
        'Debate': 52.5,
        'SC_ft': 43.2,   # Finetuned on Debate (OOD, lower)
        'DC_ft': 46.7,  # Finetuned on Debate (OOD, lower)
        'Debate_ft': 51.8,  # Finetuned on Debate (In-distribution)
    },
}

# --- Plotting Function (WITH FINETUNED BARS) ---

def plot_protocol_comparison(results):
    """
    Generates a bar chart comparing oversight protocols (SC, DC, Debate) and
    their finetuned counterparts (SC_ft, DC_ft, Debate_ft), along with
    the unassisted baseline, across different judge expertise levels (L1, L2, L3).
    """

    judge_levels = list(results.keys())  # ['L1', 'L2', 'L3']
    protocols = ['Unassisted', 'SC', 'DC', 'Debate', 'SC_ft', 'DC_ft', 'Debate_ft']
    bar_width = 0.1  # Narrower bars to fit everything
    index = np.arange(len(judge_levels))

    fig, ax = plt.subplots(figsize=(18, 7))  # Wider figure

    colors = ['gray', 'skyblue', 'lightgreen', 'coral', 'blue', 'green', 'red']

    for i, protocol in enumerate(protocols):
        values = [results[level][protocol] for level in judge_levels]
        ax.bar(index + i * bar_width, values, bar_width,
               label=f'{protocol}', color=colors[i])

    ax.set_xlabel('Judge Expertise Level')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Comparison of Oversight Protocols and Finetuned Counterparts')
    ax.set_xticks(index + 3 * bar_width)  # Adjust x-tick position
    ax.set_xticklabels(judge_levels)
    ax.legend(title="Oversight Protocol", loc='upper left', bbox_to_anchor=(1, 1)) #legend outside
    ax.set_ylim([0, 100])
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend
    plt.savefig("protocol_comparison_finetuned.png")
    plt.show()

# --- Generate the Plot ---
plot_protocol_comparison(results)