import cv2
import numpy as np





# Example usage
img1 = np.random.rand(100, 100, 1).astype(np.float32)  # Example input, replace with actual image
img2 = np.random.rand(100, 100, 1).astype(np.float32)  # Example input, replace with actual image

translation = orb_feature_matching(img1, img2)
print(f"Translation: {translation}")