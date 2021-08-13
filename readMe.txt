Hello! Welcome to Linnea Wolniewicz's REU 2021 summer project with MIT's Haystack Observatory.

The goal of this project was to automatically detect wave events as they impact ice shelves.

This software package reads in metadata contained in info.txt and creates spectrograms, normalized spectrograms,
clustered spectrograms (using a Gaussian Mixture Model), ellipse-fitted images, and a csv file with all the times
corresponding to the centers of the ellipses for each train and test image.

To run this program, update info.txt to contain your metadata. What each value in info.txt corresponds to is listed
both in info.txt and below. Make sure your directory is formatted so that the folders 
clustered_images, images_for_openCV, normalizedPower_images, openCV_results, and rawPower_images exist.
Then run runner.py, which will automatically call functions in functions.py

To change values such as the number of fast fourier transforms (nfft) or the minimum area for an ellipse to be identified
you shall have to directly adjust the values in the function definitions in functions.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Metadata in info.txt:

Network, 
station, 
location, 
channel, 
sampling rate (Hz), 
number of days in stream to analyze, 
number of total days in current stream, 
minimum frequency (Hz), 
maximum frequency (Hz), 
number of clusters, 
% of powers to put into a training set (rest goes into the test set),
start day (in UTC Time: aka 'YYYY-MM-DDTHH:MM:SS'), 
load data from IRIS (1 = yes and 0 = no it is already loaded), 
save the GMM (1 = yes and 0 = no)