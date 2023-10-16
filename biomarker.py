import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import ast  # For safely evaluating the string
import mne
from mne.preprocessing import ICA
from mne.filter import filter_data

plt.ion()  # Turn on interactive mode

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

# Convert sessions into numpy arrays
long_blink_array = np.array(long_blink_sessions)
short_blink_array = np.array(short_blink_sessions)

# Filter the EEG data
# Typically, for EEG, a 1-50 Hz bandpass filter is applied.
sfreq = 250  # assuming a sampling frequency of 250 Hz
fmin, fmax = 1, 50  # frequency band
long_blink_array_filtered = filter_data(long_blink_array.T, sfreq, fmin, fmax).T
short_blink_array_filtered = filter_data(short_blink_array.T, sfreq, fmin, fmax).T

# Define a function to plot linked sessions for each channel
def plot_linked_sessions(data_array, title):
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))  # 4 channels

    for channel_idx in range(4):  # loop over 4 channels
        # Plotting for blink data
        blink_data = np.concatenate([data_array[session_idx * 4 + channel_idx] for session_idx in range(20)])  # link first 5 sessions for the current channel
        axs[channel_idx].plot(blink_data)
        axs[channel_idx].set_title(f"Channel {channel_idx + 1}")
        axs[channel_idx].set_xticks([])  # remove x-ticks for clarity

    axs[0].set_ylabel('EEG Value')
    axs[1].set_ylabel('EEG Value')
    axs[2].set_ylabel('EEG Value')
    axs[3].set_ylabel('EEG Value')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle(title)
    plt.draw()  # Use draw instead of show

# Call the functions to plot
plot_linked_sessions(long_blink_array, "EEG Data Linked Sessions - Long Blink")
plot_linked_sessions(short_blink_array, "EEG Data Linked Sessions - Short Blink")

# Optionally, you can put this at the end to block execution until all figures are closed
plt.show(block=True)