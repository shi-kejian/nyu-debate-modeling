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
    "Average Debate Length (tokens)": (None, 243, 117),
    "Debate Completion Rate": (None, 0.75, 0.62)
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
plot_accuracy_comparison()
plot_quality_improvement()
plot_debate_deterioration()