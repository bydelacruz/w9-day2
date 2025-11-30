import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Generate synthetic training data
# X: 2 features, 100 samples
X = np.random.rand(100, 2) * 5
# y: Binary target based on a simple linear threshold
y = (X[:, 0] + X[:, 1] > 5).astype(int)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Save the model
model_path = "models/baseline.joblib"
joblib.dump(model, model_path)

print(f"Model trained on {len(X)} samples and saved to {model_path}")
print("Test prediction for [1.5, 2.3]:", model.predict_proba([[1.5, 2.3]])[0][1])
