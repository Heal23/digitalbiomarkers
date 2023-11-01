import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import ast  # For safely evaluating the string
import mne
from scipy.signal import find_peaks
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


plt.ion()  # Turn on interactive mode

# Load the CSV files
long_blink_data = pd.read_csv('C:/dev/digitalbiomarkers/EEG-data/LongBlink.csv')
short_blink_data = pd.read_csv('C:/dev/digitalbiomarkers/EEG-data/ShortBlink.csv')

# Display data tables
print("\nTable for LongBlink.csv:")
print(tabulate(long_blink_data.head(), headers='keys', tablefmt='grid'))

print("\nTable for ShortBlink.csv:")
print(tabulate(short_blink_data.head(), headers='keys', tablefmt='grid'))

# Convert string representations to actual lists
long_blink_data['data'] = long_blink_data['data'].apply(ast.literal_eval)
short_blink_data['data'] = short_blink_data['data'].apply(ast.literal_eval)

# Flatten the data
long_blink_values = [value for sublist in long_blink_data['data'].tolist() for value in sublist]
short_blink_values = [value for sublist in short_blink_data['data'].tolist() for value in sublist]

# Chunk the data
long_blink_sessions = [long_blink_values[i:i + 510] for i in range(0, len(long_blink_values), 510)]
short_blink_sessions = [short_blink_values[i:i + 510] for i in range(0, len(short_blink_values), 510)]

# Convert to arrays
long_blink_array = np.array(long_blink_sessions)
short_blink_array = np.array(short_blink_sessions)

# Apply bandpass filter
def apply_bandpass_filter(data, l_freq=0.1, h_freq=5.0):
    sfreq = 215.0
    data_filtered = mne.filter.filter_data(data, sfreq, l_freq, h_freq, method='iir', verbose=False)
    return data_filtered

long_blink_array_filtered = apply_bandpass_filter(long_blink_array)
short_blink_array_filtered = apply_bandpass_filter(short_blink_array)

# Plot sessions function
def plot_linked_sessions(data_array, title):
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    for channel_idx in range(4):
        blink_data = np.concatenate([data_array[session_idx * 4 + channel_idx] for session_idx in range(1)])
        axs[channel_idx].plot(blink_data)
        axs[channel_idx].set_title(f"Channel {channel_idx + 1}")
        axs[channel_idx].set_xticks([])
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle(title)
    plt.draw()

# Plot the data
plot_linked_sessions(long_blink_array, "EEG Data Linked Sessions - Long Blink (Non-filtered)")
plot_linked_sessions(short_blink_array, "EEG Data Linked Sessions - Short Blink (Non-filtered)")
plot_linked_sessions(long_blink_array_filtered, "EEG Data Linked Sessions - Long Blink (Filtered)")
plot_linked_sessions(short_blink_array_filtered, "EEG Data Linked Sessions - Short Blink (Filtered)")

plt.show(block=True)




# Feature Extraction function
def extract_features_per_channel(data_array):
    # Initialize feature lists
    features_per_channel = {f"Channel_{i+1}": [] for i in range(4)}
    
    for session_idx in range(len(data_array) // 4):  # 4 channels per session
        for channel_idx in range(4):
            session = data_array[session_idx * 4 + channel_idx]
            
            # Time domain features
            
            # 1. Peak-to-peak amplitude
            positive_peaks, _ = find_peaks(session)
            negative_peaks, _ = find_peaks(-session)
            if positive_peaks.size and negative_peaks.size:
                max_peak = max(session[positive_peaks])
                min_peak = min(session[negative_peaks])
                peak_to_peak_amplitude = max_peak - min_peak
            else:
                peak_to_peak_amplitude = 0
            
            # 2. Mean
            mean_val = np.mean(session)
            
            # 3. Variance
            variance_val = np.var(session)
            
            # Frequency domain features
            
            # Compute the Power Spectral Density
            f, Pxx = welch(session, fs=250.0, nperseg=256)
            
            # 4. Dominant frequency
            dominant_frequency = f[np.argmax(Pxx)]
            
            # 5. Bandwidth
            half_power = np.max(Pxx) / 2
            indices = np.where(Pxx > half_power)
            first_index = indices[0][0]
            last_index = indices[0][-1]
            bandwidth = f[last_index] - f[first_index]
            
            # Append features to the channel's feature list
            features_per_channel[f"Channel_{channel_idx+1}"].append([peak_to_peak_amplitude, mean_val, variance_val, dominant_frequency, bandwidth])

    # Convert lists to arrays
    for key in features_per_channel:
        features_per_channel[key] = np.array(features_per_channel[key])

    return features_per_channel

long_blink_features_per_channel = extract_features_per_channel(long_blink_array_filtered)
short_blink_features_per_channel = extract_features_per_channel(short_blink_array_filtered)

# Compute average features function
def compute_average_features_per_channel(features_dict):
    average_features = {}
    for key in features_dict:
        avg_peak_to_peak_amplitude = np.mean(features_dict[key][:, 0])
        avg_mean_val = np.mean(features_dict[key][:, 1])
        avg_variance_val = np.mean(features_dict[key][:, 2])
        avg_dominant_frequency = np.mean(features_dict[key][:, 3])
        avg_bandwidth = np.mean(features_dict[key][:, 4])
        
        average_features[key] = [avg_peak_to_peak_amplitude, avg_mean_val, avg_variance_val, avg_dominant_frequency, avg_bandwidth]
    
    return average_features

# Calculate average features for long and short blinks
long_blink_avg_features_per_channel = compute_average_features_per_channel(long_blink_features_per_channel)
short_blink_avg_features_per_channel = compute_average_features_per_channel(short_blink_features_per_channel)

# Print results
print("\nAverage Features for Long Blink:")
for key in long_blink_avg_features_per_channel:
    print(f"{key}:")
    print(f"Peak-to-Peak Amplitude: {long_blink_avg_features_per_channel[key][0]}")
    print(f"Mean: {long_blink_avg_features_per_channel[key][1]}")
    print(f"Variance: {long_blink_avg_features_per_channel[key][2]}")
    print(f"Dominant Frequency: {long_blink_avg_features_per_channel[key][3]} Hz")
    print(f"Bandwidth: {long_blink_avg_features_per_channel[key][4]} Hz\n")

print("Average Features for Short Blink:")
for key in short_blink_avg_features_per_channel:
    print(f"{key}:")
    print(f"Peak-to-Peak Amplitude: {short_blink_avg_features_per_channel[key][0]}")
    print(f"Mean: {short_blink_avg_features_per_channel[key][1]}")
    print(f"Variance: {short_blink_avg_features_per_channel[key][2]}")
    print(f"Dominant Frequency: {short_blink_avg_features_per_channel[key][3]} Hz")
    print(f"Bandwidth: {short_blink_avg_features_per_channel[key][4]} Hz\n")


   # Preparing Dataset

# 1. Label the data
# 0 for short blink, 1 for long blink
labels_short = np.zeros(short_blink_features_per_channel["Channel_1"].shape[0])
labels_long = np.ones(long_blink_features_per_channel["Channel_1"].shape[0])

# 2. Merge data from both blink types
all_features = []
for channel in ["Channel_1", "Channel_2", "Channel_3", "Channel_4"]:
    all_features_channel = np.vstack([short_blink_features_per_channel[channel], long_blink_features_per_channel[channel]])
    all_features.append(all_features_channel)

all_features = np.hstack(all_features)  # Combining features from all channels
all_labels = np.hstack([labels_short, labels_long])

# 3. Split dataset 70/30
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)


print("Dataset successfully split into training and testing sets.")

"""# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred)) """

# Convert the labels to one-hot encoding
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Build a simple neural network model
model = Sequential()

# Add input layer (assuming X_train has shape (num_samples, num_features))
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

# Add hidden layer
model.add(Dense(64, activation='relu'))

# Add output layer
model.add(Dense(2, activation='softmax'))  # 2 classes: short and long blink

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=50, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_categorical)

# Predict on the test set
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Print results
print("\nModel Performance:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# store the model (dump thwe model) 


