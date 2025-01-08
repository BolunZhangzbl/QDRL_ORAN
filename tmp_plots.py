import os
import numpy as np
import matplotlib.pyplot as plt


def raised_cosine_spectrum(freq, W, beta):
    """
    Generate a raised cosine spectrum.
    :param freq: Frequency range (array)
    :param W: Bandwidth (Hz or MHz)
    :param beta: Roll-off factor (0 to 1)
    :return: Spectrum values (array)
    """
    spectrum = np.zeros_like(freq)
    for i, f in enumerate(freq):
        if abs(f) <= (1 - beta) * W:
            spectrum[i] = 1  # Flat portion
        elif (1 - beta) * W < abs(f) <= (1 + beta) * W:
            spectrum[i] = 0.5 * (1 + np.cos(np.pi * (abs(f) - (1 - beta) * W) / (beta * W)))

    spectrum[(freq < -W) | (freq > W)] = 0  # Mask outside [-W, W]
    return spectrum


def plot_raised_cosine_spectra(W, fs, beta):
    """
    Plot raised cosine spectra for multiple sampled frequencies and their sum.

    :param W: Bandwidth (Hz or MHz)
    :param fs: List of sampled frequencies (array-like, Hz or MHz)
    :param beta: Roll-off factor (0 to 1)
    """

    # Frequency range for plotting (adjusted to a broader range for clarity)
    freq_range = np.linspace(-3 * W, 3 * W, 1000)

    # Generate and sum spectra
    total_spectrum = np.zeros_like(freq_range)
    individual_spectra = []

    for center_freq in fs:
        shifted_freq = freq_range - center_freq  # Shift spectrum to the center frequency
        spectrum = raised_cosine_spectrum(shifted_freq, W, beta)
        individual_spectra.append(spectrum)
        total_spectrum += spectrum

    # Plot the individual spectra and total spectrum
    plt.figure(figsize=(10, 6))

    for i, spectrum in enumerate(individual_spectra):
        plt.plot(freq_range, spectrum, label=f"Spectrum {i + 1} (Center={fs[i]} MHz)")

    # Plot the total spectrum (sum of all individual spectra)
    plt.plot(freq_range, total_spectrum, 'k--', label="Total Spectrum (Sum)", linewidth=1.5)

    # Optional: Mark the edges of the bandwidth (-W and W)
    plt.axvline(-W, color="red", linestyle="--", label=f"-W = {-W} MHz")
    plt.axvline(W, color="red", linestyle="--", label=f"W = {W} MHz")

    # Add labels, title, and legend
    plt.title(f"Raised Cosine Spectrum with Multiple Components (Beta={beta})", fontsize=18)
    plt.xlabel("Frequency (MHz)", fontsize=24)
    plt.ylabel("Amplitude", fontsize=24)
    plt.ylim([-0.1, 2.1])
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    dir_base = r"C:\Users\13580\Downloads\week 9\figures"
    file_name = f"spectrum_fs{fs[-1]}_beta{beta}.png"
    file_path = os.path.join(dir_base, file_name)
    plt.savefig(file_path, format='png', dpi=300)

    plt.show()


# Example usage of the function
fs = 1
values = [0, 0.2, 0.4, 0.6, 0.8, 1]
for beta_val in values:
    plot_raised_cosine_spectra(W=2, fs=[-fs, 0, fs], beta=beta_val)
