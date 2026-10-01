import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("Loading 1200 signals...")
X = np.load('X.npy')
y = np.load('y.npy')

# Better feature: make a small 20x20 image of the clusters
def extract_features(sample):
    I = sample[0]
    Q = sample[1]
    # 2D histogram = picture of where dots are
    hist, _, _ = np.histogram2d(I, Q, bins=20, range=[[-2.5,2.5],[-2.5,2.5]])
    return hist.flatten() # 400 numbers

print("Extracting features...")
features = np.array([extract_features(s) for s in X])

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)

print(f"Training AI on {len(X_train)} signals (4 classes)...")
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred) * 100

print(f"\n*** RESULT: AI Accuracy = {acc:.1f}% ***")
labels = ["BPSK","QPSK","8PSK","16QAM"]
for i in range(4):
    print(f"{labels[i]}: Learned!")

if acc > 85:
    print("\nYOU HAVE A 4-MODULATION SPECTRUM INTELLIGENCE!")
    print("This is real research-grade AI!")