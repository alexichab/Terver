import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# read the contents of the uploaded files to understand the data format
with open('all-in.txt', 'r') as file_in, open('all-out.txt', 'r') as file_out:
    data_in = file_in.readlines()
    data_out = file_out.readlines()

# Displaying the first few lines of each file to understand their structure
data_in[:5], data_out[:5]

# Converting the data into numpy arrays
data_in_array = np.array([int(line.strip()) for line in data_in])
data_out_array = np.array([int(line.strip()) for line in data_out])

# Calculating the autocorrelation function for both series up to a lag of 2000
acf_in = acf(data_in_array, nlags=2000, fft=True)
acf_out = acf(data_out_array, nlags=2000, fft=True)

# Plotting the autocorrelation functions
plt.figure(figsize=(15, 6))

# Autocorrelation plot for input data
plt.subplot(1, 2, 1)
plt.plot(acf_in, marker='o', linestyle='-', color='blue')
plt.title('Autocorrelation Function - Input Data')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.xlim(0, 2000)

# Autocorrelation plot for output data
plt.subplot(1, 2, 2)
plt.plot(acf_out, marker='o', linestyle='-', color='red')
plt.title('Autocorrelation Function - Output Data')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.xlim(0, 2000)
plt.tight_layout()
plt.show()