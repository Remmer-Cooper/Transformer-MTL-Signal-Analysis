import os
import json
import matplotlib.pyplot as plt

# Set up the output directory
json_folder = "./results/"  # Your JSON files directory
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

# Initialize dictionaries 
modulation_data = {}
signal_data = {}
snr_list = None

# Handle each JSON file
for file in json_files:
    with open(os.path.join(json_folder, file), 'r') as f:
        data = json.load(f)
        config = str(data['config'])  
        snrs = list(map(float, data['snr_modulation_acc'].keys()))
        snrs.sort()

        
        mod_acc = [data['snr_modulation_acc'][str(snr)] for snr in snrs]
        sig_acc = [data['snr_signal_acc'][str(snr)] for snr in snrs]

        modulation_data[config] = mod_acc
        signal_data[config] = sig_acc

        if snr_list is None:
            snr_list = snrs

# ----------modulation---------
plt.figure(figsize=(8, 5))
for config, accs in modulation_data.items():
    plt.scatter(snr_list, accs, marker='o', label=f"{config}")

plt.xlabel("Signal-to-Noise-Ratio (SNR) dB")
plt.ylabel("Modulation Classification Accuracy")
plt.title("Modulation vs. SNR")
plt.legend(title="Config")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/fig11_modulation_snr_acc.png", dpi=300)

# ---------- signal ---------
plt.figure(figsize=(8, 5))
for config, accs in signal_data.items():
    plt.scatter(snr_list, accs, marker='o', label=f"{config}")

plt.xlabel("Signal-to-Noise-Ratio (SNR) dB")
plt.ylabel("Signal Classification Accuracy")
plt.title("Signal vs. SNR")
plt.legend(title="Config")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/fig12_signal_snr_acc.png", dpi=300)

plt.show()
