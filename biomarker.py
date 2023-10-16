import pandas as pd

# Load the CSV files
long_blink_data = pd.read_csv('C:/dev/digitalbiomarkers/EEG-data/LongBlink.csv')
short_blink_data = pd.read_csv('C:/dev/digitalbiomarkers/EEG-data/ShortBlink.csv')

# Display the content of the CSV files
print("Content of LongBlink.csv:")
print(long_blink_data)

print("\nContent of ShortBlink.csv:")
print(short_blink_data)
