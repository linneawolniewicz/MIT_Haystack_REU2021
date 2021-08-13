import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.dates as mdates
import numpy as np
import obspy
import datetime
import cv2 as cv

plt.rcParams.update({'font.size': 12})

def get_power(data, sample_rate=1.0, nfft=1024, freq_min=0.0, freq_max=0.5):
    """
    This function returns the normalized and raw power of the data values using plt.specgram

    Args:
        data (NumPy array): Timeseries data
        sample_rate (float): Sampling rate (Hz)
        nfft (int): Number of fast fourier transforms (powers of 2)
        freq_min (float): Minimum frequency in Hz
        freq_max (float): Maximum frequency in Hz
    """
    plt.figure()
    power, freqs, times, im = plt.specgram(data, Fs = sample_rate, NFFT=nfft, mode='psd', scale='dB', cmap='jet', scale_by_freq=True) 
    plt.close()

    # Reduce range to freq_min-freq_max
    power = power[(freqs >= freq_min) & (freqs <= freq_max)]
    freqs = freqs[(freqs >= freq_min) & (freqs <= freq_max)]

    # Get difference spectrum
    mean_timeBins = 10*np.log10(np.median(power, axis=0))
    mean_frequencyBins = 10*np.log10(np.median(power, axis=1))
    power = 10*np.log10(power)

    # Normalize power
    power_norm_freq = np.zeros(power.shape)
    power_norm_freq_time = np.zeros(power.shape)

    for i in range(power.shape[0]): 
        value = power[i, :]
        power_norm_freq[i, :] = value - mean_frequencyBins[i]

    return(power, power_norm_freq, freqs, times)

def visualize_kmeans(power, freqs, times, freq_min, freq_max, starttime, path, filename, type=1):
    """
    This function saves a spectrogram using input data

    Args:
        power (NumPy array): Power associated with freqs and times
        freqs (NumPy array): Array of frequency values
        times (NumPy array): Array of time values (in seconds)
        freq_min (float): Minimum frequency in Hz
        freq_max (float): Maximum frequency in Hz
        starttime (UTCDateTime object): Start time of the stream
        path (String): Path to save figure
        filename (String): Filename for saved figure
        type (Integer): If 0 plots raw power. If 1 plots normalized power. If 2 plots clustered image. Only colorbar
                        limits, labels, and visibility are changed by this toggle.
    """
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()

    # Convert times array from seconds to python datetime object
    time_values = []
    for i in(range(len(times))):
        time = obspy.UTCDateTime(starttime + times[i]).datetime
        time_values.append(time)

    # Plot
    im = plt.pcolormesh(time_values, freqs, power, shading='auto', cmap='inferno')
    plt.ylim(freq_min, freq_max)
    plt.ylabel("Frequency (Hz)")

    # Add colorbar depending on type passed in
    if (type == 0): 
        cbar = plt.colorbar(im)
        cbar.set_label("Power (dB re 1 m^2/Hz)")
        im.set_clim(vmin=-180, vmax=-40)
    elif (type ==1): 
        cbar = plt.colorbar(im)
        cbar.set_label("Difference Spectrum (dB)")
        im.set_clim(vmin=-20, vmax=20)

    # Format Time Axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    fig.autofmt_xdate()

    # Save
    plt.savefig(path + filename)
    plt.close()

    return()

def create_image(power, freqs, times, freq_min, freq_max, path, filename):
    """
    This function creates a spectrogram image (no axes) using input data

    Args:
        power (NumPy array): Power associated with freqs and times
        freqs (NumPy array): Array of frequency values
        times (NumPy array): Array of time values
        freq_min (float): Minimum frequency in Hz
        freq_max (float): Maximum frequency in Hz
        path (String): Path to save figure
        filename (String): Filename for saved figure
    """
    fig = plt.figure(figsize=(14, 4))

    # Turn off axes and remove whitespace/padding
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Plot
    plt.pcolormesh(times, freqs, power, shading='auto', cmap='inferno')
    plt.ylim(freq_min, freq_max)

    # Save
    plt.savefig(path + filename)
    plt.close()

    return()

def findEllipses(edges, min_area=300, max_area=10000, min_vert_diff=1.5):
    """
    Finds ellipses in an 'edged' image using OpenCV

    Args:
        edges (image): An 'edged' image returned from the cv.Canny function 
        min_area (Int): Minimum area (in # pixels^2) of a detected ellipse
        max_area (Int): Maximum area (in # pixels^2) of a detected ellipse
        min_vert_diff (Float): Minimum angle difference (in degrees) from vertical of a detected ellipse
    """

    # Get contoured image
    contours, _ = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    ellipseMask = np.zeros(edges.shape, dtype=np.uint8)
    contourMask = np.zeros(edges.shape, dtype=np.uint8)
    centers = []

    # For each contour, find ellipses
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)

        # Skip contour if wrong size for fitEllipse function or if empty
        if len(contour) < 5: continue
        if area < 10: continue

        # Fit an ellipse
        (x, y), (MA, ma), angle = cv.fitEllipseDirect(contour)
        area_ellipse = (np.pi/4)*MA*ma

        # Skip this ellipse if it is small or vertical
        if ((area_ellipse <= min_area) or (area_ellipse >= max_area)): continue
        if (((360-min_vert_diff) < angle < 360) or (0 < angle < min_vert_diff) or ((180-min_vert_diff) < angle < (180+min_vert_diff))): continue

        # Add ellipse to ellipseMask
        poly = cv.ellipse2Poly((int(x), int(y)), (int(MA/2), int(ma/2)), int(angle), 0, 360, 5)
        hull = cv.convexHull(contour)

        cv.fillPoly(ellipseMask, [poly], 255)
        cv.fillPoly(contourMask, [hull], 255)
        centers.append(x)

    return ellipseMask, contourMask, centers

def plot_ellipses(filename_im, path, filename_save, starttime, N, min_area=300, max_area=10000, min_vert_diff=1.5, blur_kernel=19, min_thresh=100):
    """
    This function finds ellipses in an image and plots them

    Args:
        filename_im (String): Filename of the image to detect ellipses on
        path (String): Path to save figure
        filename_save (String): Filename for saved figure
        starttime (UTCDateTime object): Start time of the stream
        N (Int): Number of days in stream
        min_area (Int): Minimum area (in # pixels^2) of a detected ellipse
        max_area (Int): Maximum area (in # pixels^2) of a detected ellipse
        min_vert_diff (Float): Minimum angle difference (in degrees) from vertical of a detected ellipse
        blur_kernel (Int): Size of the kernel used for blurring the image. Must be an odd integer
        min_thresh (Int): Minimum threshold for detecting edges in cv.Canny
    """

    # Load and blur image
    im = cv.imread(path + filename_im)
    im_blur = cv.medianBlur(im, blur_kernel)

    # Detect edges
    edges = cv.Canny(im_blur, min_thresh, 3*min_thresh, 3)

    # Find ellipses
    ellipseMask, contourMask, centers = findEllipses(edges, min_area, max_area, min_vert_diff)

    # Convert centers from pixel numbers to datetime objects
    shape = im.shape
    num_pixels = shape[1]
    N_seconds = N * 24 * 60 * 60
    seconds_per_pixel = int(N_seconds / num_pixels)
    centers_datetime = []

    for i in range(len(centers)):
        value = starttime + (centers[i] * seconds_per_pixel)
        value_string = value.datetime.strftime("%Y-%m-%d-%H-%M-%S-%f")
        centers_datetime.append(value_string)

    # Plot and save images
    fig, axes = plt.subplots(1, 3, figsize=(6, 1))

    axes[0].imshow(im, cmap='inferno')
    axes[0].set_title("Original image")
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)

    axes[1].imshow(im_blur, cmap='inferno')
    axes[1].set_title("Blurred image")
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)

    axes[2].imshow(ellipseMask, cmap='inferno')
    axes[2].set_title("Ellipse image")
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.03, hspace=0.01)
    plt.savefig(path + filename_save)

    plt.close()

    return(centers_datetime)