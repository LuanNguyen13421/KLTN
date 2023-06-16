import numpy as np
import cv2

def extract_features_histogram(image, BIN):
    # Compute the color histogram for each channel
    hist_r = cv2.calcHist([image], [0], None, [BIN], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [BIN], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [BIN], [0, 256])
    
    # Normalize the histograms
    hist_r = cv2.normalize(hist_r, hist_r)
    hist_g = cv2.normalize(hist_g, hist_g)
    hist_b = cv2.normalize(hist_b, hist_b)

    feature_vector = np.concatenate((hist_r, hist_g, hist_b))
    feature_vector = np.squeeze(feature_vector)
    return feature_vector
