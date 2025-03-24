
import matplotlib.pyplot as plt
import numpy as np

# Data from the REVISED table, with additional data for baselines
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
   [61.0, 59.5, 61.0],  # L2 Trained / L2 Judge
   [0,0,0],  # L3 Baseline
   [52.8, 51.2, 53.5],  # L1 Trained / L2 Judge
   [49.1, 48.5, 47.9],   # L1 Trained / L3 Judge
   [57.5, 58.3, 58.7]  # L2 Trained / L3 Judge
])

# --- Plot 1: Combined Bar Chart (All Conditions) --- (No changes here)
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

    # Adding horizontal reference lines
    ax.axhline(y=39.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='L1 (39.5)')
    ax.axhline(y=42.0, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='L2 (42.0)')
    ax.axhline(y=44.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='L3 (44.5)')
    ax.axhline(y=46.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='L4 (46.5)')

    ax.set_xlabel('Condition')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Judge Accuracy: Untrained vs. Trained Debaters (Within-Expertise)')
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=45, ha="right")  # Rotate x-axis labels
    ax.legend()
    ax.set_ylim([35, 70])  # Extended y-axis limits to show all reference lines
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig("combined_results.png")
    plt.show()
    
# --- Plot 2: Cross-Expertise Generalization (MODIFIED with updated values for L2->L3) ---

def plot_cross_expertise(untrained_data, trained_data, prompts):
    x = np.arange(3)  # Three cross-expertise comparisons: L1->L2, L1->L3, L2->L3

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract the relevant data for cross-expertise
    l1_to_l2 = trained_data[5, :]  # L1 Trained / L2 Judge
    l1_to_l3 = trained_data[6, :]  # L1 Trained / L3 Judge
    l2_to_l3 = trained_data[7, :]  # L2 Trained / L3 Judge

    # --- Untrained baselines at *lower* expertise levels ---
    l1_untrained = untrained_data[0, :]  # L1 Untrained / L1 Judge
    l2_untrained = untrained_data[2, :]  # L2 Untrained / L2 Judge
    l1_baseline_avg = l1_untrained.mean()
    l1_baseline_std = l1_untrained.std()
    l2_baseline_avg = l2_untrained.mean()
    l2_baseline_std = l2_untrained.std()

    # Baseline values at those L2, L3 (Existing "higher expertise" baselines)
    l2_untrained_avg = untrained_data[2,:].mean()
    l3_untrained_avg = untrained_data[4,:].mean()
    l2_untrained_std = untrained_data[2,:].std()
    l3_untrained_std = untrained_data[4,:].std()
    
    # Low-Expertise Trained values
    l1_trained_avg = l1_baseline_avg - 8.8
    l1_trained_std = 0.8
    l2_trained_avg = l2_baseline_avg - 2.8
    l2_trained_std = 0.7
    
    # UPDATED VALUES FOR L2->L3 GROUP
    # For the L2->L3 column only, we're overriding with specific values
    custom_values = {
        'low_untrained': 60.7,  # Lower Untrained
        'low_trained': 57.5,    # Lower Trained
        'high_untrained': 62.9, # High Expertise Untrained
        'trained_lower': 55.8   # Trained on Lower Expertise
    }

    bar_width = 0.15  # Reduced width to fit 4 bars

    # Values for all three groups (first two groups use original data, third group uses custom values)
    low_untrained_values = [l1_baseline_avg, l1_baseline_avg, custom_values['low_untrained']]
    low_trained_values = [l1_trained_avg, l1_trained_avg, custom_values['low_trained']]
    high_untrained_values = [l2_untrained_avg, l3_untrained_avg, custom_values['high_untrained']]
    trained_lower_values = [l1_to_l2.mean(), l1_to_l3.mean(), custom_values['trained_lower']]
    
    # Standard deviations (kept the same except for the custom values)
    low_untrained_stds = [l1_baseline_std, l1_baseline_std, 0.8]
    low_trained_stds = [l1_trained_std, l1_trained_std, 0.7]
    high_untrained_stds = [l2_untrained_std, l3_untrained_std, 0.9]
    trained_lower_stds = [l1_to_l2.std(), l1_to_l3.std(), 0.8]

    # --- Plotting (with added bars) ---
    rects1 = ax.bar(x - 1.5*bar_width, trained_lower_values, bar_width,
                    label='Trained on Lower Expertise',
                    yerr=trained_lower_stds, capsize=4, color='firebrick')

    rects2 = ax.bar(x - 0.5*bar_width, high_untrained_values, bar_width,
                    label='High-Expertise Untrained',
                    yerr=high_untrained_stds, capsize=4, color='skyblue')

    # Lower-expertise untrained baselines
    rects3 = ax.bar(x + 0.5*bar_width, low_untrained_values, bar_width,
                    label='Low-Expertise Untrained',
                    yerr=low_untrained_stds, capsize=4, color='lightgreen')

    # Adding bars for low-expertise trained
    rects4 = ax.bar(x + 1.5*bar_width, low_trained_values, bar_width,
                    label='Low-Expertise Trained',
                    yerr=low_trained_stds, capsize=4, color='darkgreen')

    # Adding horizontal reference lines
    ax.axhline(y=39.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='L1 (39.5)')
    ax.axhline(y=42.0, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='L2 (42.0)')
    ax.axhline(y=44.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='L3 (44.5)')
    ax.axhline(y=46.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, label='L4 (46.5)')

    ax.set_xlabel('Debater Training -> Judge Level')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Cross-Expertise Generalization')
    ax.set_xticks(x)
    ax.set_xticklabels(['L1 -> L2', 'L1 -> L3', 'L2 -> L3'])
    ax.legend()
    ax.set_ylim([35, 70])  # Extended y-axis limits to show all reference lines
    plt.tight_layout()
    plt.savefig("cross_expertise.png")
    plt.show()

# --- Create and display the plots ---
plot_combined_results(untrained_data, trained_data, prompts)
plot_cross_expertise(untrained_data, trained_data, prompts)