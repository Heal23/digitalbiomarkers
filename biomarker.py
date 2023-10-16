import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import ast  # For safely evaluating the string

# Load the CSV files
long_blink_data = pd.read_csv('C:/dev/digitalbiomarkers/EEG-data/LongBlink.csv')
short_blink_data = pd.read_csv('C:/dev/digitalbiomarkers/EEG-data/ShortBlink.csv')

# Display LongBlink.csv data as a table
print("\nTable for LongBlink.csv:")
print(tabulate(long_blink_data.head(), headers='keys', tablefmt='grid'))

# Display ShortBlink.csv data as a table
print("\nTable for ShortBlink.csv:")
print(tabulate(short_blink_data.head(), headers='keys', tablefmt='grid'))

# Convert string representations to actual lists
long_blink_data['data'] = long_blink_data['data'].apply(ast.literal_eval)
short_blink_data['data'] = short_blink_data['data'].apply(ast.literal_eval)

# Flatten the data into single long lists
long_blink_values = [value for sublist in long_blink_data['data'].tolist() for value in sublist]
short_blink_values = [value for sublist in short_blink_data['data'].tolist() for value in sublist]

# Break the data into chunks of 510 (sessions)
long_blink_sessions = [long_blink_values[i:i + 510] for i in range(0, len(long_blink_values), 510)]
short_blink_sessions = [short_blink_values[i:i + 510] for i in range(0, len(short_blink_values), 510)]

# 1.2 Data Description

# Compute Average amplitude and standard deviation
avg_amplitudes_long = np.mean(long_blink_sessions, axis=1)
avg_amplitudes_short = np.mean(short_blink_sessions, axis=1)

std_devs_long = np.std(long_blink_sessions, axis=1)
std_devs_short = np.std(short_blink_sessions, axis=1)

print(f"Average amplitude for Long Blink sessions: {np.mean(avg_amplitudes_long):.2f}")
print(f"Average amplitude for Short Blink sessions: {np.mean(avg_amplitudes_short):.2f}")
print(f"Standard deviation for Long Blink sessions: {np.mean(std_devs_long):.2f}")
print(f"Standard deviation for Short Blink sessions: {np.mean(std_devs_short):.2f}")

# Frequency distribution (Fourier Transform) for the first session as an example
sampling_rate = 250
frequencies = np.fft.rfftfreq(510, d=1./sampling_rate)  # For a session of length 510
fft_values_long = np.fft.rfft(long_blink_sessions[0])
fft_values_short = np.fft.rfft(short_blink_sessions[0])

plt.figure(figsize=(10, 5))
plt.plot(frequencies, np.abs(fft_values_long)**2, label='Long Blink')
plt.plot(frequencies, np.abs(fft_values_short)**2, label='Short Blink')
plt.title("Power Spectral Density of the First Session")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim([0, 60])
plt.legend()
plt.show()

# Continue with your plotting function

def plot_eeg_sessions_on_single_plot(sessions, title_prefix):
    # Create a large figure with multiple subplots
    fig, axs = plt.subplots(len(sessions), 1, figsize=(10, len(sessions) * 1.5))
    fig.suptitle(title_prefix + ' EEG Sessions', y=1.02)
    
    # Loop over each session and plot it on its respective subplot
    for idx, session in enumerate(sessions):
        axs[idx].plot(session)
        
        # Set the y-axis label to be the session title
        axs[idx].set_ylabel(f'Session {idx + 1}', rotation=0, labelpad=30, verticalalignment='center', fontsize=10)
        
        # Remove x-axis labels and ticks for clarity
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        
    # Adjust spacing between plots
    plt.tight_layout()

# Plot sessions for LongBlink.csv and ShortBlink.csv, only the first 20 sessions
plot_eeg_sessions_on_single_plot(long_blink_sessions[:20], 'Long Blink')
plot_eeg_sessions_on_single_plot(short_blink_sessions[:20], 'Short Blink')

plt.show()
