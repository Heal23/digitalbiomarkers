import pandas as pd
import numpy as np
import mne
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


"""# Statistical Summary
print("Statistics for LongBlink.csv:")
print(long_blink_data.describe())
print("\nStatistics for ShortBlink.csv:")
print(short_blink_data.describe())

# Missing values
print("Missing values in LongBlink.csv:", long_blink_data.isnull().sum())
print("Missing values in ShortBlink.csv:", short_blink_data.isnull().sum())


# Visualise distribution
plt.figure(figsize=(10,5))
long_blink_data['data'].apply(ast.literal_eval).explode().astype(float).hist(alpha=0.5, label='Long Blink')
short_blink_data['data'].apply(ast.literal_eval).explode().astype(float).hist(alpha=0.5, label='Short Blink')
plt.legend()
plt.xlabel('EEG Value')
plt.ylabel('Frequency')
plt.title('Distribution of EEG Values')
plt.show()"""


# Convert string representations to actual lists
long_blink_data['data'] = long_blink_data['data'].apply(ast.literal_eval)
short_blink_data['data'] = short_blink_data['data'].apply(ast.literal_eval)

# Flatten the data into single long lists
long_blink_values = [value for sublist in long_blink_data['data'].tolist() for value in sublist]
short_blink_values = [value for sublist in short_blink_data['data'].tolist() for value in sublist]

# Break the data into chunks of 510 (sessions)
long_blink_sessions = [long_blink_values[i:i + 510] for i in range(0, len(long_blink_values), 510)]
short_blink_sessions = [short_blink_values[i:i + 510] for i in range(0, len(short_blink_values), 510)]

# Plot EEG data on a single figure with subplots
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

# Plot sessions for LongBlink.csv and ShortBlink.csv, only the first 10 sessions
plot_eeg_sessions_on_single_plot(long_blink_sessions[:10], 'Long Blink')
plot_eeg_sessions_on_single_plot(short_blink_sessions[:10], 'Short Blink')

plt.show()

