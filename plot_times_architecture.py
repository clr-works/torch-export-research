import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
file_name = "inject_inference_times.csv"
data = pd.read_csv(file_name)

# Calculate avg values
avg_inject_time = data['Inject Time (s)'].mean()
avg_inference_time = data['Inference Time (s)'].mean()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data['Sample Index'], data['Inject Time (s)'], label='Inject Time (s)', marker='o')
plt.plot(data['Sample Index'], data['Inference Time (s)'], label='Inference Time (s)', marker='s')

plt.axhline(y=avg_inject_time, color='blue', linestyle='--', label=f'Avg Inject Time: {avg_inject_time:.2f} s')
plt.axhline(y=avg_inference_time, color='orange', linestyle='--', label=f'Avg Inference Time: {avg_inference_time:.2f} s')

plt.xlabel('Sample Index')
plt.ylabel('Time (s)')
plt.title('Inject Time and Inference Time per Sample IBM Granite 3 2B')
plt.legend()  # Correctly display the legend
plt.grid()

plt.savefig("inject_vs_inference_times_with_averages.png", dpi=300)

