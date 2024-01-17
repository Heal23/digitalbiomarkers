import pandas as pd
import numpy as np
import mne
from scipy.signal import find_peaks, welch
import ast
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to apply a bandpass filter to EEG data
def apply_bandpass_filter(data, l_freq=0.1, h_freq=5.0):
    sfreq = 215.0
    return mne.filter.filter_data(data, sfreq, l_freq, h_freq, method='iir', verbose=False)

# Function to extract features from EEG data
def extract_features_per_channel(data_array):
    features_per_channel = {f"Channel_{i+1}": [] for i in range(4)}
    for session_idx in range(len(data_array)):
        for channel_idx in range(4):
            session = data_array[session_idx, channel_idx, :]
            
            # Time domain features
            positive_peaks, _ = find_peaks(session)
            negative_peaks, _ = find_peaks(-session)
            peak_to_peak_amplitude = max(session[positive_peaks]) - min(session[negative_peaks]) if positive_peaks.size and negative_peaks.size else 0
            mean_val = np.mean(session)
            variance_val = np.var(session)
            
            # Frequency domain features
            f, Pxx = welch(session, fs=250.0, nperseg=256)
            dominant_frequency = f[np.argmax(Pxx)]
            half_power = np.max(Pxx) / 2
            indices = np.where(Pxx > half_power)
            bandwidth = f[indices[0][-1]] - f[indices[0][0]] if indices[0].size > 0 else 0
            
            # Append features
            features_per_channel[f"Channel_{channel_idx+1}"].append([peak_to_peak_amplitude, mean_val, variance_val, dominant_frequency, bandwidth])

    for key in features_per_channel:
        features_per_channel[key] = np.array(features_per_channel[key])

    return features_per_channel


# Load the trained Decision Tree model
model_path = 'C:/dev/digitalbiomarkers/decision_tree_model.joblib'
model = joblib.load(model_path)

# Load new data for short and long blinks
new_long_data_path = 'C:\dev\digitalbiomarkers\EEG-data\LongBlink_test.csv'
new_short_data_path = 'C:\dev\digitalbiomarkers\EEG-data\ShortBlink_test.csv'

new_long_data = pd.read_csv(new_long_data_path)
new_short_data = pd.read_csv(new_short_data_path)
new_long_data['data'] = new_long_data['data'].apply(ast.literal_eval)
new_short_data['data'] = new_short_data['data'].apply(ast.literal_eval)

# Data processing and feature extraction
new_long_values = [value for sublist in new_long_data['data'].tolist() for value in sublist]
new_short_values = [value for sublist in new_short_data['data'].tolist() for value in sublist]

# Reshape the data into sessions
new_long_sessions = np.array(new_long_values).reshape(-1, 4, 510)  # Adjust 510 based on your session length
new_short_sessions = np.array(new_short_values).reshape(-1, 4, 510)

# Apply bandpass filter and extract features
new_long_array_filtered = apply_bandpass_filter(new_long_sessions)
new_long_features = extract_features_per_channel(new_long_array_filtered)
new_short_array_filtered = apply_bandpass_filter(new_short_sessions)
new_short_features = extract_features_per_channel(new_short_array_filtered)

# Combine features for prediction
all_new_features = np.hstack([np.vstack([new_short_features[channel], new_long_features[channel]]) for channel in ["Channel_1", "Channel_2", "Channel_3", "Channel_4"]])

# Predict using the Decision Tree model
predicted_classes = model.predict(all_new_features)

# Output predictions
print("Predicted Classes:", predicted_classes)

# If true labels are available
true_labels = np.concatenate([np.zeros(len(new_short_features['Channel_1'])), np.ones(len(new_long_features['Channel_1']))])
accuracy = accuracy_score(true_labels, predicted_classes)
conf_matrix = confusion_matrix(true_labels, predicted_classes)
class_report = classification_report(true_labels, predicted_classes)

print("\nModel Evaluation on New Data:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
