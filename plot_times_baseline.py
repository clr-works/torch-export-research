import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
file_name = "export_inference_times.csv"  # Efile in same dir
data = pd.read_csv(file_name) # not handling file not found error

# Calculate averages
avg_export_time = data['Export Time (s)'].mean()
avg_inference_time = data['Inference Time (s)'].mean()

# plot
plt.figure(figsize=(10, 6))
plt.plot(data['Sample Index'], data['Export Time (s)'], label='Export Time (s)', marker='o')
plt.plot(data['Sample Index'], data['Inference Time (s)'], label='Inference Time (s)', marker='s')

# Add average lines
plt.axhline(avg_export_time, color='blue', linestyle='--', label=f'Avg Export Time ({avg_export_time:.2f}s)')
plt.axhline(avg_inference_time, color='orange', linestyle='--', label=f'Avg Inference Time ({avg_inference_time:.2f}s)')

plt.xlabel('Sample Index')
plt.ylabel('Time (s)')
plt.title('Export Time vs Inference Time per Sample IBM Granite 3 2B')
plt.legend('i need to add a legend')
plt.grid()

plt.savefig("export_vs_inference_times.png", dpi=300)


