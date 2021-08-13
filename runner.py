import os
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
import functions
import obspy
import csv
from obspy.clients.fdsn import Client

path = os.path.dirname(__file__)
powers = []
raw_powers = []
centers_test = []
centers_train = []
names_test = []
names_train = []

# Load info.txt with all of the metadata
metadata = np.loadtxt(path + '/info.txt', dtype='str', comments='#', delimiter=',')

# Get variables from metadata
net = metadata[0]
sta = metadata[1]
loc = metadata[2]
chan = metadata[3]
sample_rate = float(metadata[4])
N = int(metadata[5])
M = int(metadata[6])
freq_min = float(metadata[7])
freq_max = float(metadata[8])
num_clusters = int(metadata[9])
train_test_split = float(metadata[10])
start_day = obspy.UTCDateTime(metadata[11])
load_data = int(metadata[12])
save_gmm = int(metadata[13])

data_filename = sta + '_' + chan + '.mseed'

# If load_data = 1, load data
if load_data == 1:
    client = Client("IRIS")
    endtime = start_day + ((M + 1) * 60 * 60 * 24)

    # Load data
    st = client.get_waveforms(net, sta, loc, chan, start_day, endtime, attach_response=True)

    # Remove instrument response
    st_rem = st.copy()
    st_rem = st_rem.remove_response(output="DISP", plot=False)

    # Save stream
    st_rem.write(path + '/' + data_filename, format='MSEED')

"""
------------------------------------------------------------------------------------------------
Load and prepare data for K-means
------------------------------------------------------------------------------------------------
"""

# Get normalized power for every N day interval in total M days
idx = 1
while ((idx+N) < M):
    # Load data
    starttime = start_day + (idx * 60 * 60 * 24)
    endtime = starttime + ((N+1) * 60 * 60 * 24)

    st = obspy.core.stream.read(path + '/' + data_filename)
    st.trim(starttime, endtime)

    data = st[0].data
    times = st[0].times()

    # Get power data
    raw_power, power, freqs, times = functions.get_power(data, sample_rate, freq_min=freq_min, freq_max=freq_max)

    # Append power_reshape to powers array for Kmeans
    powers.append(power)
    raw_powers.append(raw_power)

    # Update i by 1
    idx += 1

"""
------------------------------------------------------------------------------------------------
Train and test GMM
------------------------------------------------------------------------------------------------
"""

# Split powers into a training and test set
powers = np.array(powers)
raw_powers = np.array(raw_powers)

idx_split = int(train_test_split*powers.shape[0])

powers_train = powers[:idx_split, :, :]
powers_test = powers[idx_split:, :, :]

raw_powers_train = raw_powers[:idx_split, :, :]
raw_powers_test = raw_powers[idx_split:, :, :]

# Reshape powers for Kmeans
powers_reshape_train = powers_train.reshape(powers_train.shape[0] * powers_train.shape[1] * powers_train.shape[2], 1)
powers_reshape_test = powers_test.reshape(powers_test.shape[0] * powers_test.shape[1] * powers_test.shape[2], 1)

# Train Kmeans/Gaussian Mixture on the first few image in powers
gm = GaussianMixture(n_components=num_clusters, init_params='kmeans').fit(powers_reshape_train)

# If save_gmm = 1, save the model
if save_gmm == 1:
    # Save the model
    filename = '/gaussianMixture_' + sta + '.sav'
    pickle.dump(gm, open(path + filename, 'wb'))

# Test the model
clustered_train = gm.means_[gm.predict(powers_reshape_train)]
clustered_test = gm.means_[gm.predict(powers_reshape_test)]

# Reshape clustered power
clustered_train_reshape = clustered_train.flatten().reshape(powers_train.shape[0], powers_train.shape[1], powers_train.shape[2])
clustered_test_reshape = clustered_test.flatten().reshape(powers_test.shape[0], powers_test.shape[1], powers_test.shape[2])

# Get only event cluster from clustered_reshape
event_cluster = np.argmax(gm.means_)
clustered_train_reshape_event = np.where(clustered_train_reshape == gm.means_[event_cluster], clustered_train_reshape, 0)
clustered_test_reshape_event = np.where(clustered_test_reshape == gm.means_[event_cluster], clustered_test_reshape, 0)

"""
------------------------------------------------------------------------------------------------
Visualize GMM results
------------------------------------------------------------------------------------------------
"""

# Plot all training images
for i in  range(powers_train.shape[0]):
    # Get time of image
    starttime = start_day + (i * 60 * 60 * 24)
    name = sta + '_train_' + str(i).zfill(2) + '.jpg'
    names_train.append(name[:-4])

    # Plot raw power image
    filename = '/rawPower_images/' + name
    functions.visualize_kmeans(raw_powers_train[i, :, :], freqs, times, freq_min, freq_max, starttime, path, filename, type=0)

    # Plot normalized power image
    filename = '/normalizedPower_images/' + name
    functions.visualize_kmeans(powers_train[i, :, :], freqs, times, freq_min, freq_max, starttime, path, filename, type=1)

    # Plot K-means
    filename = '/clustered_images/' + name
    functions.visualize_kmeans(clustered_train_reshape[i, :, :], freqs, times, freq_min, freq_max, starttime, path, filename, type=2)

    # Plot K-means for OpenCV
    filename_im = '/images_for_openCV/' + name
    functions.create_image(clustered_train_reshape_event[i, :, :], freqs, times, freq_min, freq_max, path, filename_im)

    # Find ellipses in image
    filename = '/openCV_results/' + name
    filename_csv = '/center_times.csv'
    center = functions.plot_ellipses(filename_im, path, filename, starttime, N)
    centers_train.append(center)

# Plot all test images
for i in  range(powers_test.shape[0]):
    # Get time of image
    starttime = start_day + ((i + idx_split) * 60 * 60 * 24)
    name = sta + '_test_' + str(i).zfill(2) + '.jpg'
    names_test.append(name[:-4])

    # Plot raw power image
    filename = '/rawPower_images/' + name
    functions.visualize_kmeans(raw_powers_test[i, :, :], freqs, times, freq_min, freq_max, starttime, path, filename, type=0)

    # Plot normalized power image
    filename = '/normalizedPower_images/' + name
    functions.visualize_kmeans(powers_test[i, :, :], freqs, times, freq_min, freq_max, starttime, path, filename, type=1)

    # Plot K-means
    filename = '/clustered_images/' + name
    functions.visualize_kmeans(clustered_test_reshape[i, :, :], freqs, times, freq_min, freq_max, starttime, path, filename, type=2)

    # Plot K-means for OpenCV
    filename_im = '/images_for_openCV/' + name
    functions.create_image(clustered_test_reshape_event[i, :, :], freqs, times, freq_min, freq_max, path, filename_im)

    # Find ellipses in image
    filename = '/openCV_results/' + name
    filename_csv = '/center_times.csv'
    center = functions.plot_ellipses(filename_im, path, filename, starttime, N)
    centers_test.append(center)

"""
------------------------------------------------------------------------------------------------
Save centers to a file
------------------------------------------------------------------------------------------------
"""

# Write centers to a file
file = open(path + '/center_times.csv', 'w', newline='')
writer = csv.writer(file, delimiter=',')

# Write all train centers to the file
for i in range(len(names_train)):
    row = [names_train[i]] + centers_train[i]
    writer.writerow(row)

# Write all test centers to the file
for i in range(len(names_test)):
    row = [names_test[i]] + centers_test[i]
    writer.writerow(row)

file.close()