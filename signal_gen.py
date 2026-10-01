import numpy as np
import matplotlib.pyplot as plt

samples = 1000
# BPSK
bpsk = np.random.choice([-1, 1], samples)
# QPSK
qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
qpsk = np.random.choice(qpsk_symbols, samples)
# Add noise
noise = 0.2 * (np.random.randn(samples) + 1j*np.random.randn(samples))
bpsk_noisy = bpsk + noise
qpsk_noisy = qpsk + noise

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(bpsk_noisy.real, bpsk_noisy.imag, s=2)
plt.title('BPSK - 2 clusters')

plt.subplot(1,2,2)
plt.scatter(qpsk_noisy.real, qpsk_noisy.imag, s=2)
plt.title('QPSK - 4 clusters')

plt.tight_layout()
plt.savefig('first_signals.png')
print("DONE!")
plt.show()
