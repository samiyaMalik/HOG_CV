!pip install opencv-python numpy scipy matplotlib

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

def get_differential_filter():
    """Generate 1D differential filters for x and y directions."""
    filter_x = np.array([[-1, 0, 1]])  # Horizontal gradient
    filter_y = np.array([[-1], [0], [1]])  # Vertical gradient
    return filter_x, filter_y

def filter_image(image, filter):
    """Apply convolution filter to the image."""
    return signal.convolve2d(image, filter, mode='same', boundary='symm')

def get_gradient(im_dx, im_dy):
    """Calculate gradient magnitude and angle."""
    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    grad_angle = np.arctan2(im_dy, im_dx)
    grad_angle = np.mod(grad_angle, np.pi)
    return grad_mag, grad_angle

def build_histogram(grad_mag, grad_angle, cell_size):
    """Build histogram of oriented gradients for each cell."""
    M, N = grad_mag.shape
    n_cells_x = N // cell_size
    n_cells_y = M // cell_size
    n_bins = 9
    bin_width = np.pi / n_bins

    hist = np.zeros((n_cells_y, n_cells_x, n_bins))

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = grad_mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_angle = grad_angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            for m in range(cell_size):
                for n in range(cell_size):
                    angle = cell_angle[m, n]
                    mag = cell_mag[m, n]

                    bin_idx = int(angle // bin_width)
                    bin_idx_prev = (bin_idx - 1) % n_bins
                    bin_idx_next = (bin_idx + 1) % n_bins

                    ratio = (angle % bin_width) / bin_width

                    hist[i, j, bin_idx_prev] += mag * ratio * (1 - ratio)
                    hist[i, j, bin_idx] += mag * (1 - ratio)
                    hist[i, j, bin_idx_next] += mag * ratio * ratio

    return hist

def get_block_descriptor(hog_cell, block_size=2):
    """Normalize HOG descriptors for each block."""
    n_cells_y, n_cells_x, n_bins = hog_cell.shape
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1

    hog_descriptor = []

    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            block = hog_cell[i:i+block_size, j:j+block_size, :].ravel()
            norm = np.sqrt(np.sum(block**2) + 1e-6)
            normalized_block = block / norm
            hog_descriptor.append(normalized_block)

    return np.concatenate(hog_descriptor)

def extract_hog(im, cell_size=8, block_size=2, fixed_size=64):
    """Extract HOG features from the input image."""
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    im = cv2.resize(im, (fixed_size, fixed_size))
    im = im.astype(float) / 255.0

    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)

    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    hog_cell = build_histogram(grad_mag, grad_angle, cell_size)
    hog_descriptor = get_block_descriptor(hog_cell, block_size)

    return hog_descriptor

def face_recognition(image, template_hog, cell_size=8, block_size=2, template_size=64, stride=4):
    """Perform face recognition using template matching with stride."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    gray = cv2.resize(gray, (256, 256))
    gray = gray.astype(float) / 255.0
    M, N = gray.shape

    scores = np.zeros(((M - template_size) // stride + 1, (N - template_size) // stride + 1))

    for i in range(0, M - template_size + 1, stride):
        for j in range(0, N - template_size + 1, stride):
            patch = gray[i:i+template_size, j:j+template_size]
            patch_hog = extract_hog(patch, cell_size, block_size, fixed_size=template_size)
            score = np.dot(patch_hog, template_hog) / (
                np.sqrt(np.sum(patch_hog**2)) * np.sqrt(np.sum(template_hog**2)) + 1e-6)
            scores[i // stride, j // stride] = score

    return scores, gray

def visualize_hog(im, hog_descriptor, cell_size=8, block_size=2):
    """Visualize HOG features."""
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(im, cmap='gray')
    plt.title('Input Image')

    plt.subplot(122)
    n_cells_y = im.shape[0] // cell_size - block_size + 1
    n_cells_x = im.shape[1] // cell_size - block_size + 1
    plt.imshow(hog_descriptor.reshape(n_cells_y, n_cells_x, -1).mean(axis=2), cmap='hot')
    plt.title('HOG Features')
    plt.colorbar()
    plt.show()

def visualize_face_detection(image, scores):
    """Visualize face detection results."""
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')

    plt.subplot(122)
    plt.imshow(scores, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Detection Scores')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load sample image and template
    image = cv2.imread('sample_face.jpg')
    template = cv2.imread('template.png')

    # Check if images are loaded correctly
    if image is None or template is None:
        raise ValueError("Failed to load one or both images. Check file paths.")

    # Extract HOG features from template
    template_hog = extract_hog(template, fixed_size=64)

    # Perform face recognition
    scores, processed_image = face_recognition(image, template_hog, template_size=64, stride=4)

    # Visualize results
    visualize_hog(cv2.resize(cv2.cvtColor(template, cv2.COLOR_RGB2GRAY), (64, 64)), template_hog)
    visualize_face_detection(processed_image, scores)