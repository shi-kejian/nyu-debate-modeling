import matplotlib.pyplot as plt
import numpy as np

# Data from the REVISED table (organized for easier plotting)
prompts = ["Baseline", "Role-Playing", "Evidence", "Critical Error"]
l1_untrained = np.array([[54.2, 52.8, 53.9],
                        [52.5, 55.8, 51.9],
                        [56.3, 54.2, 57.1],
                        [55.8, 56.3, 56.0]])

l1_trained = np.array([[48.5, 51.7, 50.1],
                      [46.2, 50.5, 45.5],
                      [47.8, 46.5, 49.7],
                      [53.7, 54.5, 54.8]])

l2_untrained = np.array([[57.5, 53.8, 58.9],
                        [57.2, 59.8, 57.1],
                        [58.9, 60.2, 57.1],
                        [60.5, 59.1, 60.3]])

l2_trained = np.array([[49.1, 46.9, 51.0],
                      [42.1, 45.3, 41.6],
                      [41.8, 45.2, 43.9],
                      [55.2, 56.0, 54.8]])

# --- Plot 1: Bar chart comparing L1 Untrained vs. L1 Trained ---
def plot_l1_comparison(l1_untrained, l1_trained, prompts):
    bar_width = 0.35
    index = np.arange(len(prompts))

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(index - bar_width/2, l1_untrained.mean(axis=1), bar_width,
                    label='L1 Untrained', yerr=l1_untrained.std(axis=1), capsize=4)
    rects2 = ax.bar(index + bar_width/2, l1_trained.mean(axis=1), bar_width,
                    label='L1 Trained', yerr=l1_trained.std(axis=1), capsize=4)

    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Judge Accuracy on L1: Untrained vs. Trained Debaters')
    ax.set_xticks(index)
    ax.set_xticklabels(prompts)
    ax.legend()
    ax.set_ylim([40, 65]) # Set consistent y-axis limits
    plt.tight_layout()
    plt.savefig("l1_comparison.png") #save the plot
    plt.show()


# --- Plot 2: Bar chart comparing L2 Untrained vs. L2 Trained ---
def plot_l2_comparison(l2_untrained, l2_trained, prompts):
    bar_width = 0.35
    index = np.arange(len(prompts))

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(index - bar_width/2, l2_untrained.mean(axis=1), bar_width,
                    label='L2 Untrained', yerr=l2_untrained.std(axis=1), capsize=4)
    rects2 = ax.bar(index + bar_width/2, l2_trained.mean(axis=1), bar_width,
                    label='L2 Trained', yerr=l2_trained.std(axis=1), capsize=4)

    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Judge Accuracy on L2: Untrained vs. Trained Debaters')
    ax.set_xticks(index)
    ax.set_xticklabels(prompts)
    ax.legend()
    ax.set_ylim([40, 65])  #Consistent y limits.
    plt.tight_layout()
    plt.savefig("l2_comparison.png")
    plt.show()

# --- Plot 3: Line chart showing generalization gap ---

def plot_generalization_gap(l1_untrained, l1_trained, l2_untrained, l2_trained, prompts):
    index = np.arange(len(prompts))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate the mean accuracy for each condition
    l1_untrained_mean = l1_untrained.mean(axis=1)
    l1_trained_mean = l1_trained.mean(axis=1)
    l2_untrained_mean = l2_untrained.mean(axis=1)
    l2_trained_mean = l2_trained.mean(axis=1)

    ax.plot(index, l1_untrained_mean, label='L1 Untrained', marker='o', linestyle='-')
    ax.plot(index, l1_trained_mean, label='L1 Trained', marker='x', linestyle='--')
    ax.plot(index, l2_untrained_mean, label='L2 Untrained', marker='s', linestyle='-')
    ax.plot(index, l2_trained_mean, label='L2 Trained (from L1)', marker='^', linestyle='--')

    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Generalization Gap: L1 vs. L2 Performance')
    ax.set_xticks(index)
    ax.set_xticklabels(prompts)
    ax.legend()
    ax.set_ylim([40, 65])  # Consistent y limits.
    plt.tight_layout()
    plt.savefig("generalization_gap.png")
    plt.show()


# --- Plot 4: Combined Bar Chart (L1 and L2, Trained and Untrained) ---

def plot_combined_comparison(l1_untrained, l1_trained, l2_untrained, l2_trained, prompts):
    bar_width = 0.2

    index = np.arange(len(prompts))

    fig, ax = plt.subplots(figsize=(14, 7))

    rects1 = ax.bar(index - 3*bar_width/2, l1_untrained.mean(axis=1), bar_width, label='L1 Untrained', color='skyblue')
    rects2 = ax.bar(index - bar_width/2, l1_trained.mean(axis=1), bar_width, label='L1 Trained', color='royalblue')
    rects3 = ax.bar(index + bar_width/2, l2_untrained.mean(axis=1), bar_width, label='L2 Untrained', color='lightcoral')
    rects4 = ax.bar(index + 3*bar_width/2, l2_trained.mean(axis=1), bar_width, label='L2 Trained (from L1)', color='firebrick')
    
    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_title('Judge Accuracy: L1 vs. L2, Untrained vs. Trained')
    ax.set_xticks(index)
    ax.set_xticklabels(prompts)
    ax.legend()
    ax.set_ylim([40, 65])
    plt.tight_layout()
    plt.savefig("combined_comparison.png") #save the plot
    plt.show()

# --- Create and display the plots ---
plot_l1_comparison(l1_untrained, l1_trained, prompts)
plot_l2_comparison(l2_untrained, l2_trained, prompts)
plot_generalization_gap(l1_untrained, l1_trained, l2_untrained, l2_trained, prompts)
plot_combined_comparison(l1_untrained, l1_trained, l2_untrained, l2_trained, prompts)