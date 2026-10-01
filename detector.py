import numpy as np
import pickle

# Load your saved brain
with open('spectrum_ai.pkl', 'rb') as f:
    model = pickle.load(f)

labels = ["BPSK", "QPSK", "8PSK", "16QAM"]

def make_unknown_signal(mod_type):
    noise = 0.3 * (np.random.randn(800) + 1j*np.random.randn(800))
    if mod_type == 0:
        data = np.random.choice([-1, 1], 800) + noise
    elif mod_type == 1:
        symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
        data = np.random.choice(symbols, 800) + noise
    elif mod_type == 2:
        angles = np.random.choice(np.arange(8) * 2*np.pi/8, 800)
        data = np.exp(1j*angles) + noise
    else:
        I_vals = np.array([-3,-1,1,3])
        Q_vals = np.array([-3,-1,1,3])
        I = np.random.choice(I_vals, 800)
        Q = np.random.choice(Q_vals, 800)
        data = (I + 1j*Q)/np.sqrt(10)*2 + noise
    return np.array([data.real, data.imag])

def extract_features(sample):
    I, Q = sample[0], sample[1]
    hist, _, _ = np.histogram2d(I, Q, bins=20, range=[[-2.5,2.5],[-2.5,2.5]])
    return hist.flatten().reshape(1,-1)

print("=== SPECTRUM INTELLIGENCE LIVE ===\n")
for _ in range(5):
    true_type = np.random.randint(0,4)
    signal = make_unknown_signal(true_type)
    feat = extract_features(signal)
    pred = model.predict(feat)[0]
    print(f"True: {labels[true_type]} -> AI detected: {labels[pred]} {'✅' if true_type==pred else '❌'}")

print("\nYour AI is now a real-time spectrum sensor!")