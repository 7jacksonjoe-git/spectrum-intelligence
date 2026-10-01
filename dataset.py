import numpy as np

def make_sample(mod_type, n=800):
    noise = 0.3 * (np.random.randn(n) + 1j*np.random.randn(n))

    if mod_type == 0: # BPSK - 2 clusters
        data = np.random.choice([-1, 1], n) + noise

    elif mod_type == 1: # QPSK - 4 clusters
        symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
        data = np.random.choice(symbols, n) + noise

    elif mod_type == 2: # 8PSK - 8 clusters (circle)
        angles = np.random.choice(np.arange(8) * 2*np.pi/8, n)
        data = np.exp(1j*angles) + noise

    else: # 16QAM - 16 clusters (grid)
        I_vals = np.array([-3, -1, 1, 3])
        Q_vals = np.array([-3, -1, 1, 3])
        I = np.random.choice(I_vals, n)
        Q = np.random.choice(Q_vals, n)
        data = (I + 1j*Q)/np.sqrt(10) * 2 + noise

    return np.array([data.real, data.imag])

X = []
y = []
labels = {0:"BPSK", 1:"QPSK", 2:"8PSK", 3:"16QAM"}

for mod in range(4):
    print(f"Generating {labels[mod]}...")
    for i in range(300):
        X.append(make_sample(mod))
        y.append(mod)

X = np.array(X)
y = np.array(y)
np.save('X.npy', X)
np.save('y.npy', y)
print(f"\nDone! X shape: {X.shape}, y shape: {y.shape}")
print("300 x 4 = 1200 signals ready!")