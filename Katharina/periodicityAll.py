import numpy as np
import matplotlib.pyplot as plt
import pickle
import uuid
from scipy import fft, ndimage
from scipy.signal import find_peaks
from multiprocessing import Pool
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

sample = 50
size = 100  # size of the 2D grid
T = 30
lib_sizes = [100]

c_list = [0,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
 0.11, 0.12, 0.13, 0.14]


########### We can use two lines below to get part of results
#sample = 2
#c_list = [0, 0.04, 0.08,0.12]


a1 = np.linspace(2.8e-3, 2.8e-5,3)
a2 = np.flip(a1)
a_list = np.dstack((a1,a2)).squeeze()

tasks = len(c_list)*a_list.shape[0]

dx = 2. / size  # space step

def detect_spatial_periodicity_matrix(input_matrix,
                                      remove_trend=True,
                                      gaussian_sigma=10,
                                      peak_threshold_std=2,
                                      window_type='hanning',
                                      visualize=True,
                                      figsize=(15, 10)):
    """
    Detecting Spatial Periodicity in Matrix Data Using 2D Fourier Transform


    Parameters:
    input_matrix: Input 2D numpy array
    remove_trend: Whether to remove the trend component
    gaussian_sigma: Sigma value for Gaussian filtering, used for trend estimation
    peak_threshold_std: Threshold for peak detection (standard deviation multiplier)
    window_type: Window function type, options include 'hanning', 'hamming', 'blackman'
    visualize: Whether to generate visualizations
    figsize: Figure size

    Returns:
    results: Dictionary containing analysis results
    """

    # 1. Input validation and preprocessin
    if not isinstance(input_matrix, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if input_matrix.ndim != 2:
        raise ValueError("Input must be a 2D array")

    print(f"Input matrix dimensions: {input_matrix.shape}")

    # Copy the original matrix to avoid modifying the source data
    matrix = input_matrix.astype(np.float64).copy()

    # 2. Data preprocessing
    print("Performing data preprocessing...")

    # Remove trend component (optional)
    if remove_trend:
        # Estimate trend using 2D Gaussian filtering
        trend = ndimage.gaussian_filter(matrix, sigma=gaussian_sigma)
        detrended_matrix = matrix - trend
        print("Trend component removed")
    else:
        detrended_matrix = matrix.copy()
        print("Trend component not removed")

    # Apply window function to reduce edge effects
    rows, cols = detrended_matrix.shape

    if window_type == 'hanning':
        window_row = np.hanning(rows)
        window_col = np.hanning(cols)
    elif window_type == 'hamming':
        window_row = np.hamming(rows)
        window_col = np.hamming(cols)
    elif window_type == 'blackman':
        window_row = np.blackman(rows)
        window_col = np.blackman(cols)
    else:
        window_row = np.ones(rows)
        window_col = np.ones(cols)

    window_2d = window_row[:, np.newaxis] * window_col[np.newaxis, :]
    windowed_matrix = detrended_matrix * window_2d

    # 3. 2D Fourier Transform
    print("Performing 2D Fourier transform...")

    # Perform FFT
    fft_result = fft.fft2(windowed_matrix)

    # Shift zero frequency to center
    fft_shifted = fft.fftshift(fft_result)

    # Calculate power spectral density
    power_spectrum = np.abs(fft_shifted) ** 2

    # Take logarithm of power spectrum for better visualization
    log_power_spectrum = np.log10(power_spectrum + 1e-10)  # 避免log(0)

    # 4. Detect periodic features
    print("Analyzing periodic features...")

    # Get matrix center
    center_row, center_col = rows // 2, cols // 2

    #  Create distance matrix (distance from center)
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    r = np.sqrt(x ** 2 + y ** 2)

    #  Calculate radially averaged power spectrum
    radial_power = []
    max_radius = min(center_row, center_col)
    radii = np.arange(1, max_radius)

    for radius in radii:
        mask = (r >= radius - 0.5) & (r < radius + 0.5)
        if np.any(mask):
            radial_power.append(np.mean(power_spectrum[mask]))
        else:
            radial_power.append(0)

    radial_power = np.array(radial_power)

    #  Find significant peaks (possible periods)
    if len(radial_power) > 0:
        # Smooth radial power spectrum for better peak detection
        window_size = min(5, len(radial_power) // 10)
        if window_size % 2 == 0:
            window_size += 1

        if window_size > 1:
            smoothed_radial = np.convolve(radial_power, np.ones(window_size) / window_size, mode='same')
        else:
            smoothed_radial = radial_power.copy()

        # Find peaks
        height_threshold = np.mean(smoothed_radial) + peak_threshold_std * np.std(smoothed_radial)
        peaks, properties = find_peaks(smoothed_radial, height=height_threshold, distance=max_radius // 20)
    else:
        peaks = np.array([])
        smoothed_radial = np.array([])

    # 5. Calculate spatial periods
    spatial_periods = []
    peak_frequencies = []

    if len(peaks) > 0:
        for peak in peaks:
            if peak < len(radial_power):
                # Calculate normalized frequency (between 0 and 0.5)
                frequency = peak / (max_radius*2)
                if frequency > 0:
                    # Calculate spatial period (in pixels)
                    spatial_period = 1.0 / frequency if frequency > 1e-10 else float('inf')
                    spatial_periods.append(spatial_period)
                    peak_frequencies.append(frequency)
                    print(f"Detected spatial period: {spatial_period:.2f} pixels  (frequency: {frequency:.4f})")

    # 6. Directionality analysis
    print("Performing directionality analysis...")

    # Calculate angular distribution
    angles = np.arctan2(y, x) * 180 / np.pi
    angle_bins = np.linspace(-180, 180, 37)  # 每10度一个bin
    angular_power = []
    angular_std = []

    for i in range(len(angle_bins) - 1):
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        if np.any(mask):
            angular_power.append(np.mean(power_spectrum[mask]))
            angular_std.append(np.std(power_spectrum[mask]))
        else:
            angular_power.append(0)
            angular_std.append(0)

    angular_power = np.array(angular_power)
    angular_std = np.array(angular_std)

    # Find dominant direction
    if len(angular_power) > 0:
        dominant_angle_idx = np.argmax(angular_power)
        dominant_angle = (angle_bins[dominant_angle_idx] + angle_bins[dominant_angle_idx + 1]) / 2
        print(f"Detected dominant direction: {dominant_angle:.1f} degree")
    else:
        dominant_angle = None

    peak_to_mean_ratio = np.max(radial_power) / np.mean(radial_power)

    print(f"peak_to_mean_ratio: {peak_to_mean_ratio:.2f} ")


    # 7. Result summary
    results = {
       # 'original_matrix': matrix,
       # 'detrended_matrix': detrended_matrix,
        # 'power_spectrum': power_spectrum,
        # 'log_power_spectrum': log_power_spectrum,
        #  'radial_power': radial_power,
        # 'smoothed_radial': smoothed_radial,
        'peaks': peaks,
        'spatial_periods': spatial_periods,
        'peak_frequencies': peak_frequencies,
        'angular_power': angular_power,
        'angular_std': angular_std,
        'angle_bins': angle_bins,
        'dominant_angle': dominant_angle,
        'max_radius': max_radius
    }

    # 8. Visualization
    if visualize:
        print("Generating visualizations...")
        visualize_periodicity_results(results, figsize)

    return results


def visualize_periodicity_results(results, figsize=(15, 10)):
    """
     Visualize periodicity analysis results
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Original matrix
    im0 = axes[0, 0].imshow(results['original_matrix'], cmap='viridis')
    axes[0, 0].set_title('Origin Image')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])

    # Detrended matrix
    im1 = axes[0, 1].imshow(results['detrended_matrix'], cmap='viridis')
    axes[0, 1].set_title('Image detrended')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])

    # Power spectrum (log scale)
    im2 = axes[0, 2].imshow(results['log_power_spectrum'], cmap='hot')
    axes[0, 2].set_title('Power Spectrum (log scale)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])

    # Central region of power spectrum (zoomed)
    center_row = results['original_matrix'].shape[0] // 2
    center_col = results['original_matrix'].shape[1] // 2
    center_size = min(center_row, center_col) // 4

    center_region = results['log_power_spectrum'][
                    center_row - center_size:center_row + center_size,
                    center_col - center_size:center_col + center_size
                    ]
    im3 = axes[1, 0].imshow(center_region, cmap='hot')
    axes[1, 0].set_title('Central Region of Power Spectrum')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])

    # Radially averaged power spectru
    axes[1, 1].plot(results['radial_power'], alpha=0.7, label='Origin')
    if hasattr(results, 'smoothed_radial') and len(results['smoothed_radial']) > 0:
        axes[1, 1].plot(results['smoothed_radial'], 'r-', linewidth=2, label='Smoothed')

    if len(results['peaks']) > 0:
        axes[1, 1].plot(
            results['peaks'],
            results['smoothed_radial'][results['peaks']],
            'ro', markersize=8, label='detected peak'
        )

    axes[1, 1].set_xlabel('spatial frequency')
    axes[1, 1].set_ylabel('Average Power')
    axes[1, 1].set_title('Radially Averaged Power')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Directionality analysis
    angles_center = (results['angle_bins'][:-1] + results['angle_bins'][1:]) / 2
    axes[1, 2].bar(angles_center, results['angular_power'], width=10, alpha=0.7)
    axes[1, 2].set_xlabel('angle(degree)')
    axes[1, 2].set_ylabel('angular_power')
    axes[1, 2].set_title('angular distribution')
    axes[1, 2].grid(True, alpha=0.3)

    # Mark dominant direction
    if results['dominant_angle'] is not None:
        axes[1, 2].axvline(x=results['dominant_angle'], color='red', linestyle='--',
                           label=f'dominant_angle: {results["dominant_angle"]:.1f}°')
        axes[1, 2].legend()

    plt.tight_layout()
    plt.show()


def generate_periodic_test_matrix(size=(256, 256), periods=[32, 64], amplitudes=[1.0, 0.5],
                                  noise_level=0.1, random_seed=None):
    """
    Generate test matrix with known periodicity

    Parameters:
    size: Matrix dimensions (rows, cols)
    periods: List of periods (in pixels)
    amplitudes: Amplitudes corresponding to periods
    noise_level: Noise level
    random_seed: Random seed
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    rows, cols = size
    x = np.linspace(0, 4 * np.pi, cols)
    y = np.linspace(0, 4 * np.pi, rows)
    X, Y = np.meshgrid(x, y)

    # Create base matrix
    matrix = np.zeros((rows, cols))

    # Add periodic patterns
    for period, amplitude in zip(periods, amplitudes):
        # Calculate wave numbers
        kx =  period
        ky =  period

        #  Add sine wave pattern (45 degree direction)
        pattern = amplitude * np.sin(kx * X + ky * Y)
        matrix += pattern

    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, matrix.shape)
        matrix += noise

    return matrix


def print_periodicity_summary(results):
    """
    Print summary of periodicity analysis results
    """
    print("\n" + "=" * 50)
    print("Analysis Summary of Spatial Periodicity ")
    print("=" * 50)

    print(f"Analyzed matrix dimensions: {results['original_matrix'].shape}")

    if len(results['spatial_periods']) > 0:
        print(f"\nDetected {len(results['spatial_periods'])} significant spatial periods:")
        for i, (period, freq) in enumerate(zip(results['spatial_periods'], results['peak_frequencies'])):
            print(f"  Period  {i + 1}: {period:.2f} pixels  (frequency: {freq:.4f})")
    else:
        print("\nNo significant spatial periodicity detected")

    if results['dominant_angle'] is not None:
        print(f"\nDominant direction: {results['dominant_angle']:.1f} degree")

    #  Calculate overall periodicity strength metric
    if len(results['radial_power']) > 0:
        peak_to_mean_ratio = np.max(results['radial_power']) / np.mean(results['radial_power'])
        print(f"\nPeriodicity strength metrics:")
        print(f"  Peak-to-mean ratio: {peak_to_mean_ratio:.3f}")

        if peak_to_mean_ratio > 10.0:
            print("  Periodicity strength: Strong")
        elif peak_to_mean_ratio > 5.0:
            print("  Periodicity strength: Moderate")
        else:
            print("  Periodicity strength: Weak")


def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx ** 2





def run_sim(X_in, Y_in, T, c, a1, a2, plot=True, saveas=False):
    X = X_in.copy()
    Y = Y_in.copy()

    dt = .001  # time step,
    n = int(T / dt)  # number of iterations
    if plot:
        fig, axes = plt.subplots(2, 8, figsize=(6.3, 2))
        step_plot = n // 8
    # We simulate the PDE with the finite difference
    # method.
    for i in range(n):
        # We compute the Laplacian of u and v.
        deltaX = laplacian(X)
        deltaY = laplacian(Y)
        # We take the values of u and v inside the grid.
        Xc = X[1:-1, 1:-1]
        Yc = Y[1:-1, 1:-1]

        # We update the variables.
        X[1:-1, 1:-1], Y[1:-1, 1:-1] = \
            Xc + dt * (a1 * deltaX - Xc ** 2), \
            Yc + dt * (a2 * deltaY - Yc ** 2 + c * Xc * Yc)

        # Neumann conditions: derivatives at the edges
        # are null.
        for B in (X, Y):
            B[0, :] = B[1, :]
            B[-1, :] = B[-2, :]
            B[:, 0] = B[:, 1]
            B[:, -1] = B[:, -2]

        # We plot the state of the system at
        # 9 different times.
        if plot:
            if i % step_plot == 0 and i < 8 * step_plot:
                ax1 = axes[0, i // step_plot]
                ax2 = axes[1, i // step_plot]

                #show_patterns(X, ax=ax1)
                ax1.set_title(f'${i * dt:.0f}$')
                #show_patterns(Y, ax=ax2)
                # ax.set_title(f'Y $t={i * dt:.0f}$')
                if i // step_plot == 0:
                    ax1.text(-0.25, 0.5, 'X', transform=ax1.transAxes, fontsize=11, horizontalalignment='center')
                    ax2.text(-0.25, 0.5, 'Y', transform=ax2.transAxes, fontsize=11, horizontalalignment='center')

    if saveas:
        plt.tight_layout()
        plt.savefig(saveas, bbox_inches='tight')
    return X, Y


def run_periodicity(sample, size, c, a1, a2, uuid):
    results = {}
    # filename = 'results_test/c'+str(c)+'_a1'+str(a1)+'_a2'+str(a2)+'.pkl'

    # print('running with c=', c, 'and a=', a1, a2)

    # for s in range(sample):
    for s in range(sample):
        np.random.seed(seed=s)
        X_rand = np.random.rand(size, size)
        Y_rand = np.random.rand(size, size)
        X, Y = run_sim(X_rand, Y_rand, T=T, c=c, a1=a1, a2=a2, plot=False)
        periodX = detect_spatial_periodicity_matrix(
            X,
            remove_trend=False,
            peak_threshold_std=3,  #=
            visualize=False
        )
        periodY = detect_spatial_periodicity_matrix(
            Y,
            remove_trend=False,
            peak_threshold_std=3,  # =
            visualize=False
        )

        # conv = None
        # correlation_coefficient, p_value = None, None
        results[s] = {'X': periodX,
                      'Y': periodY}

    # print('finished combination')

    return results


def run_periodicity_wrapper(params):
    return run_periodicity(*params)

def run_grid():
    with Pool() as p:
        parameter_list = [(sample, size, c, a1, a2, uuid.uuid4()) for c in c_list for (a1, a2) in a_list]
        print("len(parameter_list) = ", len(parameter_list))
        # results_list = p.starmap(run_sample, parameter_list)
        results_list = list(tqdm(p.imap(run_periodicity_wrapper, parameter_list), total=len(parameter_list)))

        meta = {parameter[5]: parameter for parameter in parameter_list}
        results_map = {parameter[5]: result for parameter, result in zip(parameter_list, results_list)}

        with open('periodicity/AllResults.pkl', 'wb') as pickle_file:
            pickle.dump({'meta': meta, 'results_map': results_map}, pickle_file)



if __name__ == "__main__":
    '''
    # Demo 1: Generate  matrix and analyze periodic
    print("=== Testing matrix analyzing ===")
    test_matrix = generate_periodic_test_matrix(
        size=(256, 256),
        periods=[4, 8],
        amplitudes=[1.0, 0.5],
        noise_level=0.2,
        random_seed=42
    )

    results = detect_spatial_periodicity_matrix(
        test_matrix,
        remove_trend=False,
        gaussian_sigma=5,
        peak_threshold_std=3,
        visualize=True
    )

    print_periodicity_summary(results)

    # Demo 2: Random matrix(compare)
    print("\n" + "=" * 50)
    print("=== Random matrix analyzing(compare) ===")
    random_matrix = np.random.normal(0, 1, (256, 256))

    results_random = detect_spatial_periodicity_matrix(
        random_matrix,
        remove_trend=False,
        peak_threshold_std=3,
        visualize=True
    )
    print_periodicity_summary(results_random)

    # Demo 3: Simulated diffusion matrix
    print("\n" + "=" * 50)
    print("=== Simulated diffusion matrix analyzing(compare) ===")

    T = 30
    a1 = 2.8e-4  # 2.8e-5
    a2 = 2.8e-3
    size = 100  # size of the 2D grid
    dx = 2. / size  # space step
    dims = np.arange(1, 9)
    lib_sizes = np.arange(10, 101, 30)
    lib_size = 100
    c = 0.15

    X_rand = np.random.rand(size, size)
    Y_rand = np.random.rand(size, size)
    X, Y = run_sim(X_rand, Y_rand, T=T, c=c, a1=a1, a2=a2, plot=False)

    results_Diffuse = detect_spatial_periodicity_matrix(
        X,
        remove_trend=False,
        peak_threshold_std=3,
        visualize=True
    )
    print_periodicity_summary(results_Diffuse)

  '''

    run_grid()