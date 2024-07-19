import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from qarray import ChargeSensedDotArray, GateVoltageComposer, WhiteNoise, LatchingModel

# defining the capacitance matrices
Cdd = [[0., 0.3], [0.3, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.2, 0.05], [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.05, 0.06]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

white_noise = WhiteNoise(
    amplitude=1e-2
)

# combining the noise models via addition
noise = white_noise

latching_model = LatchingModel(
    n_dots=2,
    p_leads=[0.5, 0.5],
    p_inter=[
        [0., 0.1],
        [0.1, 0.],
    ]
)

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.05, T=100,
    noise_model = noise,
    latching_model = latching_model
)

voltage_composer = GateVoltageComposer(model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -1, 1
vy_min, vy_max = -1, 1
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer._do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

# centering the voltage sweep on the [0, 1] - [1, 0] interdot charge transition on the side of a charge sensor coulomb peak
vg += model.optimal_Vg([1, 1, 0.55])

# calculating the output of the charge sensor and the charge state for each gate voltage
z, n = model.charge_sensor_open(vg)
vg += np.array([0.1, 0.0, 0])

z_dash, n_dash = model.charge_sensor_open(vg)



fig, axes = plt.subplots(2, 2, figsize=(10, 5))

axes[0, 0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[0, 0].set_xlabel('$Vx$')
axes[0, 0].set_ylabel('$Vy$')
axes[0, 0].set_title('$z$')

axes[0, 1].imshow(z_dash, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[0, 1].set_xlabel('$Vx$')
axes[0, 1].set_ylabel('$Vy$')
axes[0, 1].set_title('$z\'$')

# high pass filter z
z = z - gaussian_filter(z, sigma=10)
z_dash = z_dash - gaussian_filter(z_dash, sigma=10)

z = np.gradient(z, axis=0) + 1j * np.gradient(z, axis=1)
z_dash = np.gradient(z_dash, axis=0) + 1j * np.gradient(z_dash, axis=1)

axes[1, 0].imshow(np.abs(z), extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[1, 0].set_xlabel('$Vx$')
axes[1, 0].set_ylabel('$Vy$')
axes[1, 0].set_title('$z$')

axes[1, 1].imshow(np.abs(z_dash), extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[1, 1].set_xlabel('$Vx$')
axes[1, 1].set_ylabel('$Vy$')
axes[1, 1].set_title('$z\'$')



def orb_feature_matching(img1, img2):

    img1 = np.abs(img1)
    img2 = np.abs(img2)

    # Ensure the images are 2D arrays
    if img1.ndim == 3 and img1.shape[2] == 1:
        img1 = img1.squeeze(axis=2)
    if img2.ndim == 3 and img2.shape[2] == 1:
        img2 = img2.squeeze(axis=2)

    # Ensure the images are still 2D
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Images must be 2D arrays (grayscale)")

    # Ensure images are in correct data type
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (img2 * 255).astype(np.uint8)

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not find enough keypoints in one or both images.")

    # Initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find the translation
    translation_vector = np.mean(points2 - points1, axis=0)

    return translation_vector

def phase_correlation_translation(img1, img2):

    img1 = img1.squeeze()
    img2 = img2.squeeze()

    window = np.hanning(img1.shape[0])[:, None] * np.hanning(img1.shape[1])
    img1_gray = img1 * window
    img2_gray = img2 * window

    # Compute the 2D Fourier Transform of the images
    dft1 = np.fft.fft2(img1_gray)
    dft2 = np.fft.fft2(img2_gray)

    # Compute the cross-power spectrum
    eps = np.finfo(float).eps  # small number to avoid division by zero
    cross_power_spectrum = (dft1 * dft2.conj()) / (np.abs(dft1 * dft2.conj()) + eps)

    # Compute the inverse Fourier Transform to get the correlation
    correlation = np.fft.ifft2(cross_power_spectrum)
    correlation = np.abs(correlation)

    # Find the peak in the correlation
    max_loc = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Calculate the translation
    translation_y, translation_x = max_loc
    h, w = img1_gray.shape

    if translation_y > h // 2:
        translation_y -= h
    if translation_x > w // 2:
        translation_x -= w

    return translation_x, translation_y

delta_x, delta_y = phase_correlation_translation(z, z_dash)

delta_Vx = delta_x * (vx_max - vx_min) / 100
delta_Vy = delta_y * (vy_max - vy_min) / 100

print(f'The translation in Vx is {delta_Vx} and in Vy is {delta_Vy}')