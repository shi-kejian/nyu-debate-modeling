import matplotlib.pyplot as plt
import numpy as np

# Experiment Setup and Results Table
experiment_details = {
    "Fixed Math Solution Model (Supervisee)": "Gemma-2 2B-it",
    "Debater Models": "2x Qwen2-1.5B-Instruct",
    "L_N Supervisor": "Llama-3.1-8B-Instruct",
    "L_N+1 Supervisor (Ground Truth Proxy)": "Ground Truth",
    "Task": "GSM8K Math Problems",
    "Training Steps": "10K",
    "Batch Size": "32",
    "Max Tokens for Debate": "512",
    "Number of Problems": "1000"
}

performance_metrics = {
    # Format: (Before Debate, With Initial Debate, After RLHF)
    "L1 Accuracy vs L2": (34.2, 34.7, 29.3),
    "L1 Accuracy vs Ground Truth": (22.1, 20.5, 18.8),
    "Debate Quality Score": (None, 0.45, 0.25),
    "SO Improvement (vs L2)": (None, 1.8, -2.0),
    "SO Improvement (vs GT)": (None, 0.7, -1.6),
    "Average Debate Length (tokens)": (None, 745.4, 571.2),
    "Debate Completion Rate": (None, 0.8, 0.86)
}

# Plot 1: Accuracy Comparison
def plot_accuracy_comparison():
    categories = ['L1 Alone', 'Test on L2']
    baseline = [56.5, 66]
    initial = [48, 55] 
    rlhf = [41.5, 51.5]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width, baseline, width, label='Unassisted', color='#3b5bdb')
    ax.bar(x, initial, width, label='With Debate -- Initial', color='#ae3ec9')
    ax.bar(x + width, rlhf, width, label='With Debate -- DPO', color='#099268')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on top of each bar
    for i, rect in enumerate(ax.patches):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Plot 1b: Accuracy vs Debate Turns
def plot_accuracy_vs_turns():
    turns = [1, 2, 3, 4, 6]
    
    # L1 Alone results
    l1_baseline = [56.5] * len(turns)  # Constant as it's unassisted
    l1_dpo = [40.0, 42.0, 45.5, 42.5, 39.0]  # Matches turn 4 result from bar plot
    
    # Test on L2 results
    l2_baseline = [66.0] * len(turns)  # Constant as it's unassisted
    l2_dpo = [50.5, 51.0, 52.5, 47.5, 48.0]  # Matches turn 4 result from bar plot
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # L1 Alone plot
    ax1.plot(turns, l1_baseline, '--', label='Unassisted', color='#ffe3e3', marker='o')
    ax1.plot(turns, l1_dpo, '-', label='With Debate -- DPO', color='#fa5252', marker='^')
    ax1.set_xlabel('Number of Debate Turns')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('L1 Alone Accuracy vs Debate Turns')
    ax1.set_ylim(30, 70)  # Set consistent y-axis limits
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Test on L2 plot
    ax2.plot(turns, l2_baseline, '--', label='Unassisted', color='#d0bfff', marker='o')
    ax2.plot(turns, l2_dpo, '-', label='With Debate -- DPO', color='#6741d9', marker='^')
    ax2.set_xlabel('Number of Debate Turns')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test on L2 Accuracy vs Debate Turns')
    ax2.set_ylim(30, 70)  # Set consistent y-axis limits
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Plot 2: Debate Quality and SO Improvement  
def plot_quality_improvement():
    metrics = ['Debate Quality', 'SO Improvement (L2)', 'SO Improvement (GT)']
    initial = [0.65, 3.5, 1.4]
    after_rlhf = [0.45, -3.9, -3.3]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width/2, initial, width, label='Initial Debate', color='#f08c00')
    ax.bar(x + width/2, after_rlhf, width, label='After DPO', color='#ff6b6b')
    
    
    ax.set_ylabel('Score')
    ax.set_title('Debate Quality and SO Improvement')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on top of each bar
    for i, rect in enumerate(ax.patches):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:+.2f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# Plot 3: Debate Deterioration Metrics
def plot_debate_deterioration():
    steps = [0, 100, 200]
    avg_lengths = [676, 658, 671]
    completion_rates = [0.94, 0.93, 0.92]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Average Length Plot
    ax1.plot(steps, avg_lengths, marker='o', linewidth=2)
    ax1.set_ylabel('Average Debate Length (tokens)')
    ax1.set_title('Consultancy Length')
    ax1.set_ylim(128, 1024)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Completion Rate Plot
    ax2.plot(steps, completion_rates, marker='o',  linewidth=2)
    ax2.set_ylabel('Debate Completion Rate')
    ax2.set_xlabel('Training Steps')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Print detailed experiment setup
print("Experiment Setup:")
print("-----------------")
for key, value in experiment_details.items():
    print(f"{key}: {value}")

print("\nPerformance Metrics:")
print("-------------------")
for key, value in performance_metrics.items():
    if value[0] is None:
        print(f"{key}: Initial: {value[1]}, After RLHF: {value[2]}")
    else:
        print(f"{key}: Before: {value[0]}, Initial: {value[1]}, After RLHF: {value[2]}")

# Generate all plots
plot_accuracy_vs_turns()
plot_accuracy_comparison()
plot_quality_improvement()
plot_debate_deterioration()



def print_model_comparison_table():
    models = ['Llama 3.1 8B Debater', 'Qwen 2.5 7B Debater', 'Qwen 2.5 14B']
    unassisted = [66.5, 66.5, 66.5]
    assisted = [57, 58, 60] 
    assisted_trained = [53.5, 0, 0]

    # Print header
    print("+" + "-"*25 + "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*18 + "+")
    print("|{:^25}|{:^12}|{:^10}|{:^18}|".format("Debater", "Unassisted", "Assisted", "Assisted (Trained)"))
    print("+" + "-"*25 + "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*18 + "+")

    # Print data rows
    for i in range(len(models)):
        print("|{:<25}|{:^12}|{:^10}|{:^18}|".format(
            models[i], 
            unassisted[i],
            assisted[i],
            assisted_trained[i]
        ))
    
    # Print bottom border
    print("+" + "-"*25 + "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*18 + "+")

# Call the function
print_model_comparison_table()
