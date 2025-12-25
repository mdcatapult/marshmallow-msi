from scipy.stats import multivariate_normal
from skimage import morphology, measure
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ADD RANDOM BLOBS
def make_random_blobs(new_ion, tissue_labels, intens = 1, no_blobs = 3, cov_matrix = (2,2)):
    final_blob_mask = np.zeros((new_ion.shape))
    
    location = random.sample(range(len(tissue_labels[0])), no_blobs)
   
    for i in location:
        
        blob = generate_elliptical_blob(final_blob_mask.shape, center  = [tissue_labels[0][i],tissue_labels[1][i]], 
                                                   intensity =intens, cov_matrix = cov_matrix)

        final_blob_mask += blob
 
    
    # plt.imshow(final_blob_mask)
    # plt.show()
    return final_blob_mask


def get_roi_blobs(wt_mask, labeled_image, count, spread_factor = 1, intensity = 5):
    blobs = []
    for l in range(1,count+1):
        roi = np.where(labeled_image==l)
        new_ion = np.zeros((wt_mask.shape))
        new_ion[roi] = 1
        # plt.imshow(new_ion)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        new_ion = cv2.dilate(new_ion, kernel, iterations=3)
        roi = np.where(new_ion==1)

        labeled_blob = measure.label(new_ion.astype(int))
        props = measure.regionprops(labeled_blob)
        center = props[0].centroid
        # mid = int(len(roi[0])/2)

        blob_y, blob_x = roi
        # print(len(blob_y), len(blob_x))

        points = np.column_stack((blob_y,blob_x))
     
        # Estimate covariance from blob shape
        y_var = np.var(blob_y) * spread_factor
        x_var = np.var(blob_x) * spread_factor
        xy_cov = np.cov(blob_y, blob_x)[0, 1] * spread_factor

        cov_matrix = np.array([[y_var, xy_cov], [xy_cov, x_var]])
        # cov_matrix = np.cov(points.T)

        # Add small regularization to avoid singular matrix
        # cov_matrix += np.eye(2) * 1e-6

        y, x = np.mgrid[0:new_ion.shape[0], 0:new_ion.shape[1]]
        pos = np.dstack((y, x))

      
        # rv = multivariate_normal((roi[0][mid],roi[1][mid]), cov_matrix)

        rv = multivariate_normal(center, cov_matrix)
        blob = intensity * rv.pdf(pos)
        # print(len(x), len(y), len(pos), len(rv.pdf(pos)))
        # plt.plot(range(len(pos)),rv.pdf(pos))
        # plt.show()
        # print('roi intensity', intensity, blob.min(),blob.max())

        # Normalize to maintain intensity scaling
        if blob.max() > 0:
            blob = blob * (intensity / blob.max())
            # print('yes',blob.max())

        blobs.append(blob)

    return blobs

def generate_elliptical_blob(shape,
                            center: [float, float],
                            intensity: float,
                            cov_matrix: np.ndarray) -> np.ndarray:
    """
    Generate elliptical blob using multivariate normal distribution.
    
    Args:
        center: (y, x) center coordinates
        intensity: Peak intensity
        cov_matrix: 2x2 covariance matrix defining ellipse shape
        
    Returns:
        2D array representing the elliptical blob
    """
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    pos = np.dstack((y, x))
    
    rv = multivariate_normal(center, cov_matrix)
    blob = intensity * rv.pdf(pos)
    
    # print('blob',intensity,blob.min(),blob.max())

    # Normalize to maintain intensity scaling

    if blob.max() > 0:
        blob = blob * (intensity / blob.max())
        # print('yes',blob.max())
        
    return blob

def add_poisson_noise(image, scaling_factor=0.5, background_level=0.2):
    """
    Add Poisson noise to an image.
    
    Args:
        image: Input image (2D array)
        scaling_factor: Scale factor to control noise level (higher = more counts = less relative noise)
        background_level: Constant background level to add before noise
        
    Returns:
        Noisy image with Poisson statistics
    """
    # Add background level
    image_with_bg = image + background_level
    
    # Scale up for Poisson sampling (higher values = less relative noise)
    scaled_image = image_with_bg * scaling_factor
    
    # Ensure non-negative values for Poisson sampling
    scaled_image = np.maximum(scaled_image, 0)
    
    # Generate Poisson noise
    # For very high values, Poisson approaches Gaussian, so we handle this
    max_val = np.max(scaled_image)
    if max_val > 700:  # Avoid overflow in Poisson generation
        # Use Gaussian approximation for high values
        noisy_scaled = np.where(
            scaled_image > 700,
            scaled_image + np.random.normal(0, np.sqrt(scaled_image), scaled_image.shape),
            np.random.poisson(scaled_image)
        )
    else:
        noisy_scaled = np.random.poisson(scaled_image)
    
    # Scale back down
    noisy_image = noisy_scaled / scaling_factor
    
    # Remove background level to return to original scale
    final_image = noisy_image - background_level
    
    return np.maximum(final_image, 0)  # Ensure non-negative
