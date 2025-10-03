import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

json_folder = "./results_json2"  # JSON files directory
output_folder = "./figures"
os.makedirs(output_folder, exist_ok=True)

sns.set_theme(style='whitegrid') 
palette = sns.color_palette("tab10")


colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
linestyles = ['-', '--', '-.', ':']

# read JSON files
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
json_files.sort()  

def plot_accuracy(task_key_train, task_key_val, title, fig_name):
    plt.figure(figsize=(10, 6))

    for idx, file in enumerate(json_files):
        file_path = os.path.join(json_folder, file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        config = tuple(data["config"])
        train_acc = data[task_key_train]
        val_acc = data[task_key_val]
        epochs = list(range(1, len(train_acc) + 1))

        color = palette[idx % len(colors)]

        style_train = linestyles[0]
        style_val = linestyles[1]

        plt.plot(epochs, val_acc, linestyle=style_val, color=color, label=f'Val-{config}', linewidth=0.8)
        plt.plot(epochs, train_acc, linestyle=style_train, color=color, label=f'Train-{config}', linewidth=0.8)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(fontsize='small', loc='lower right')
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    plt.ylim(0.2, 0.85 if "mod" in task_key_val else 0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, fig_name), dpi=300)
    plt.close()



plot_accuracy("modulation_train_acc", "modulation_val_acc",
              "MTL training performance on modulation classification task",
              "fig5_modulation_accuracy.png")

plot_accuracy("signal_train_acc", "signal_val_acc",
              "MTL training performance on signal classification task",
              "fig6_signal_accuracy.png")
