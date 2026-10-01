import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

X = np.load('X.npy')
y = np.load('y.npy')
with open('spectrum_ai.pkl', 'rb') as f:
    model = pickle.load(f)

labels = ["BPSK","QPSK","8PSK","16QAM"]

def extract_features(sample):
    I, Q = sample[0], sample[1]
    hist, _, _ = np.histogram2d(I, Q, bins=20, range=[[-2.5,2.5],[-2.5,2.5]])
    return hist.flatten()

features = np.array([extract_features(s) for s in X])
pred = model.predict(features)
cm = confusion_matrix(y, pred)

fig = plt.figure(figsize=(12,8))
fig.suptitle('SPECTRUM INTELLIGENCE - Final Report', fontsize=14, fontweight='bold')

# 4 constellation plots
for i in range(4):
    ax = plt.subplot(2,3,i+1)
    idx = np.where(y==i)[0][0]
    ax.scatter(X[idx][0], X[idx][1], s=2, alpha=0.5)
    ax.set_title(labels[i])
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.grid(True, alpha=0.3)

# Confusion matrix
ax = plt.subplot(2,3,5)
ax.imshow(cm, cmap='Blues')
ax.set_title('Confusion Matrix')
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(labels, rotation=45)
ax.set_yticklabels(labels)
for r in range(4):
    for c in range(4):
        ax.text(c, r, cm[r,c], ha='center', va='center', fontweight='bold')

# Summary
ax = plt.subplot(2,3,6)
ax.axis('off')
acc = np.mean(pred==y)*100
ax.text(0.1, 0.8, f"Dataset: {len(y)} signals", fontsize=11)
ax.text(0.1, 0.6, f"Accuracy: {acc:.1f}%", fontsize=14, fontweight='bold', color='green')
ax.text(0.1, 0.4, "Model: MLP (128,64)\nStatus: RESEARCH-GRADE ✅\nLocation: Lagos, Oct 2026", fontsize=10)

plt.tight_layout()
plt.savefig('final_report.png', dpi=300)
print("SAVED: final_report.png!")
plt.show()