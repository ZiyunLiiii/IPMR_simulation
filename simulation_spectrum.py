import spekpy as sp
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Generate spectrum
# -----------------------
s = sp.Spek(kvp=95, th=12)
s.filter('Al', 2.5)

energies, spectrum = s.get_spectrum()
# normalize to sum = 1
weights = spectrum / np.sum(spectrum)

energies = np.asarray(energies, dtype=float)
weights = np.asarray(weights, dtype=float)

# Remove possible zero or negative bins just in case
mask = weights > 0
energies = energies[mask]
weights = weights[mask]

# Weighted mean energy
mean_E = np.sum(energies * weights) / np.sum(weights)

# Weighted std
std_E = np.sqrt(np.sum(weights * (energies - mean_E)**2) / np.sum(weights))

print(f"Mean energy = {mean_E:.3f} keV")
print(f"Std energy  = {std_E:.3f} keV")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(6,4))
plt.plot(energies, weights, linewidth=1.5)

plt.xlabel('energy (keV)')
plt.ylabel('normalized weights')
plt.title('Normalized X-ray spectrum')
plt.xlim(0, 100)
plt.ylim(0, weights.max() * 1.1)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()