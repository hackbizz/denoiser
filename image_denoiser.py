import cv2
import numpy as np
import onednn as dnn

# Load the noisy image
img = cv2.imread('C:\\Users\\Lenovo\\Dropbox\\PC\\Desktop\\sample.jpg')

# Convert the image to float32 and normalize the pixel values
img = img.astype(np.float32) / 255.0

# Create a Gaussian noise filter
noise_filter = dnn.dnn_convolution2d(
    input_shape=(1, img.shape[0], img.shape[1]),
    filter_shape=(1, 3, 3, 3),
    padding='same',
    strides=(1, 1),
    activation=None
)

# Train the filter on the noisy image
noise_filter.train(np.expand_dims(img, axis=0))

# Use the trained filter to denoise the image
denoised_img = noise_filter(np.expand_dims(img, axis=0))[0, ...]

# Save the denoised image
cv2.imwrite('C:\\Users\\Lenovo\\Dropbox\\PC\\Desktop\\denoised_image.jpg', denoised_img * 255.0)
