import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from tabulate import tabulate
import ast  # For safely evaluating the string


# Load the CSV files
long_blink_data = pd.read_csv('C:/dev/digitalbiomarkers/EEG-data/LongBlink.csv')
short_blink_data = pd.read_csv('C:/dev/digitalbiomarkers/EEG-data/ShortBlink.csv')

# Display the content of the CSV files
print("Content of LongBlink.csv:")
print(long_blink_data)

print("\nContent of ShortBlink.csv:")
print(short_blink_data)

# Display LongBlink.csv data as a table
print("\nTable for LongBlink.csv:")
print(tabulate(long_blink_data.head(), headers='keys', tablefmt='grid'))

# Display ShortBlink.csv data as a table
print("\nTable for ShortBlink.csv:")
print(tabulate(short_blink_data.head(), headers='keys', tablefmt='grid'))

# Convert string representations to actual lists
long_blink_data['data'] = long_blink_data['data'].apply(ast.literal_eval)
short_blink_data['data'] = short_blink_data['data'].apply(ast.literal_eval)

# Concatenate the first 100 lists for each dataset
long_blink_array = np.concatenate(long_blink_data['data'].iloc[:500])
short_blink_array = np.concatenate(short_blink_data['data'].iloc[:500])

# Reshape arrays to 2D (channels x time)
long_blink_array = long_blink_array.reshape(1, -1)
short_blink_array = short_blink_array.reshape(1, -1)

# Assuming a sampling rate of 250 Hz, adjust as needed
sfreq = 250
ch_names = ['EEG']
ch_types = ['eeg']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create raw MNE objects for both datasets
long_blink_raw = mne.io.RawArray(long_blink_array, info)
short_blink_raw = mne.io.RawArray(short_blink_array, info)

# Plot the raw data
long_blink_raw.plot(title='Long Blink EEG (First 100 rows)')
short_blink_raw.plot(title='Short Blink EEG (First 100 rows)')

# Add this line to keep the plots open
plt.show(block=True)