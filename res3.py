import matplotlib.pyplot as plt
import numpy as np

# Data from the REVISED table
prompts = ["L1 Baseline", "L1 Trained", "L2 Baseline", "L2 Trained", "L3 Baseline", "L1->L2", "L1->L3", "L2->L3"]
untrained_data = np.array([
    [56.1, 55.9, 56.4],  # L1 Untrained / L1 Judge
    [0,0,0], # Not applicable, place holder
    [59.2, 60.5, 58.8],  # L2 Untrained / L2 Judge
    [0,0,0], # Not applicable
    [62.1, 61.5, 62.8], # L3 Untrained / L3 Judge
    [0,0,0], # Not applicable
    [0,0,0], # Not applicable
    [0,0,0]  # Not applicable

])

trained_data = np.array([
   [0,0,0],  # Not applicable
   [57.2, 58.1, 57.5], # L1 Trained / L1 Judge
   [0,0,0],  # Not applicable
   [60.3, 61.2, 56.8],  # L2 Trained / L2 Judge
   [0,0,0],  # L3 Baseline
   [52.8, 51.2, 52.5],  # L1 Trained / L2 Judge
   [49.1, 48.5, 47.9],   # L1 Trained / L3 Judge
   [60.5, 61.1, 58.9]  # L2 Trained / L3 Judge
])

# --- Plot 1: Combined Bar Chart (All Conditions) ---
def plot_combined_results(untrained_data, trained_data, prompts):

    #Since we have placeholder, we filter them
    x = np.arange(len(prompts))
    valid_indices = [i for i, prompt in enumerate(prompts) if "->" not in prompt and "Baseline" not in prompt] #skip cross judge cases
    valid_indices_untrained = [i for i, prompt in enumerate(prompts) if "Baseline" in prompt]
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 7))  # Adjusted figure size

    # Plot untrained data
    rects1 = ax.bar(x[valid_indices_untrained] - width/2, untrained_data[valid_indices_untrained,:].mean(axis=1), width, label='Untrained', color='skyblue', yerr=untrained_data[valid_indices_untrained,:].std(axis=1), capsize=4)
    # Plot trained data
    rects2 = ax.bar(x[valid_indices] + width/2, trained_data[valid_indices,:].mean(axis=1), width, label='Trained', color='royalblue', yerr=trained_data[valid_indices,:].std(axis=1), capsize=4)


    ax.set_xlabel('Condition')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Judge Accuracy: Untrained vs. Trained Debaters (Within-Expertise)')
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=45, ha="right")  # Rotate x-axis labels
    ax.legend()
    ax.set_ylim([40, 70])  # Consistent y-axis limits
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig("combined_results.png")
    plt.show()
    
    
# --- Plot 2: Cross-Expertise Generalization ---

def plot_cross_expertise(trained_data, prompts):

    x = np.arange(3)  # Three cross-expertise comparisons: L1->L2, L1->L3, L2->L3

    fig, ax = plt.subplots(figsize=(12, 6))
    # Extract the relevant data for cross-expertise
    l1_to_l2 = trained_data[5, :]  # L1 Trained / L2 Judge
    l1_to_l3 = trained_data[6, :]  # L1 Trained / L3 Judge
    l2_to_l3 = trained_data[7, :]  # L2 Trained / L3 Judge

    bar_width = 0.2

    rects1 = ax.bar(x - bar_width, [l1_to_l2.mean(), l1_to_l3.mean(), l2_to_l3.mean()], bar_width,
                    label='Trained on Lower Expertise',
                    yerr=[l1_to_l2.std(), l1_to_l3.std(), l2_to_l3.std()], capsize=4, color='lightcoral')
    
    # Baseline values at those L2, L3
    baselines = [untrained_data[2,:].mean() , untrained_data[4,:].mean(), untrained_data[4,:].mean()] #L2 untrained average, L3 untrained average
    baseline_std = [untrained_data[2,:].std(), untrained_data[4,:].std(), untrained_data[4,:].std()]
    
    rects2 = ax.bar(x, baselines, bar_width,
                    label='High-ExpertiseUntrained',
                    yerr=baseline_std, capsize=4, color = 'skyblue')

    ax.set_xlabel('Debater Training -> Judge Level')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Cross-Expertise Generalization')
    ax.set_xticks(x)
    ax.set_xticklabels(['L1 -> L2', 'L1 -> L3', 'L2 -> L3'])
    ax.legend()
    ax.set_ylim([40, 70])  # Consistent y-axis
    plt.tight_layout()
    plt.savefig("cross_expertise.png")
    plt.show()

# --- Create and display the plots ---

plot_combined_results(untrained_data, trained_data, prompts)
plot_cross_expertise(trained_data, prompts)