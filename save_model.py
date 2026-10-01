import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X = np.load('X.npy')
y = np.load('y.npy')

def extract_features(sample):
    I, Q = sample[0], sample[1]
    hist, _, _ = np.histogram2d(I, Q, bins=20, range=[[-2.5,2.5],[-2.5,2.5]])
    return hist.flatten()

features = np.array([extract_features(s) for s in X])
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500)
model.fit(X_train, y_train)

# Save it!
with open('spectrum_ai.pkl', 'wb') as f:
    pickle.dump(model, f)

print("AI SAVED as spectrum_ai.pkl - you can now use it anytime!")