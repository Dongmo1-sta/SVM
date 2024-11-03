#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


# In[13]:


data= pd.read_csv(r"C:\Users\17063\Downloads\kddcup10_percent.csv")


# In[14]:


# Step 3: Convert Categorical Variables to Numeric (One-Hot Encoding)
categorical_columns = ['protocol_type', 'service', 'flag']
data_encoded = pd.get_dummies(data, columns=categorical_columns)


# In[15]:


# Step 4: Drop Label Column for Features
X = data_encoded.drop('label', axis=1)
y = data['label']


# In[17]:


# Step 5: Apply undersampling to address class imbalance
label_counts = y.value_counts()
min_count_safe = label_counts[label_counts > 10].min()  # Ensure at least 10 samples per class

# Create a balanced dataset by undersampling each class
balanced_data = pd.concat(
    [
        X[y == label].sample(n=min(min_count_safe, len(X[y == label])), random_state=42).assign(label=label)
        for label in label_counts.index
    ]
)

# Shuffle the balanced data
balanced_data = balanced_data.sample(frac=1, random_state=42)


# In[18]:


# Separate balanced features and labels
X_resampled = balanced_data.drop('label', axis=1)
y_resampled = balanced_data['label']


# In[19]:


# Step 6: Split Dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[23]:


# Step 7: Train Logistic Regression Model (L2 regularization by default)
logistic_model = LogisticRegression(max_iter=1)
logistic_model.fit(X_train, y_train)


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[26]:


# Step 7: Train Logistic Regression Model (L2 regularization by default)
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = logistic_model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[27]:


# Step 9: Get Feature Weights
feature_weights = pd.Series(logistic_model.coef_[0], index=X_train.columns)


# In[28]:


# Step 10: Sort by Absolute Value to Find Most Influential Features
sorted_weights = feature_weights.abs().sort_values(ascending=False)
print(f'Top influential features:\n{sorted_weights.head(10)}')


# In[29]:


# Step 11: Setup L1 and L2 Regularization and Compare Accuracies
regularization_strengths = np.logspace(-4, 4, 10)  # Regularization strengths
train_accuracies_l1, test_accuracies_l1 = [], []
train_accuracies_l2, test_accuracies_l2 = [], []


# In[30]:


# Loop through regularization strengths
for C in regularization_strengths:
    # L1 Regularization
    model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=1000)
    model_l1.fit(X_train, y_train)
    train_accuracies_l1.append(model_l1.score(X_train, y_train))
    test_accuracies_l1.append(model_l1.score(X_test, y_test))


# In[31]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[32]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[33]:


for C in regularization_strengths:
    # L1 Regularization
    model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000)
    model_l1.fit(X_train_scaled, y_train)
    
    # Store training and testing accuracies
    train_accuracies_l1.append(model_l1.score(X_train_scaled, y_train))
    test_accuracies_l1.append(model_l1.score(X_test_scaled, y_test))


# In[36]:


# Initialize lists to store accuracies
train_accuracies_l2 = []
test_accuracies_l2 = []


# In[38]:


for C in regularization_strengths:
    model_l2 = LogisticRegression(penalty='l2', solver='saga', C=C, max_iter=5000, random_state=42)
    model_l2.fit(X_train_scaled, y_train)
    
    # Store training and testing accuracies
    train_accuracies_l2.append(model_l2.score(X_train_scaled, y_train))
    test_accuracies_l2.append(model_l2.score(X_test_scaled, y_test))


# In[40]:


print(f"Length of regularization_strengths: {len(regularization_strengths)}")
print(f"Length of train_accuracies_l1: {len(train_accuracies_l1)}")
print(f"Length of test_accuracies_l1: {len(test_accuracies_l1)}")
print(f"Length of train_accuracies_l2: {len(train_accuracies_l2)}")
print(f"Length of test_accuracies_l2: {len(test_accuracies_l2)}")

# Ensure all lists are of the same length before plotting
if all(len(lst) == len(regularization_strengths) for lst in [train_accuracies_l1, test_accuracies_l1, train_accuracies_l2, test_accuracies_l2]):
    plt.figure(figsize=(10, 6))
    plt.plot(regularization_strengths, train_accuracies_l1, label="L1 Train Accuracy", marker='o')
    plt.plot(regularization_strengths, test_accuracies_l1, label="L1 Test Accuracy", marker='x')
    plt.plot(regularization_strengths, train_accuracies_l2, label="L2 Train Accuracy", marker='o')
    plt.plot(regularization_strengths, test_accuracies_l2, label="L2 Test Accuracy", marker='x')
    plt.xscale('log')
    plt.xlabel('Regularization Strength (C)')
    plt.ylabel('Accuracy')
    plt.title('Train/Test Accuracy vs Regularization Strength (L1 and L2)')
    plt.legend()
    plt.show()
else:
    print("Error: Mismatched lengths between regularization strengths and accuracy lists.")


# In[42]:


# L1 and L2 regularization loops, ensuring 20 iterations for consistency
for C in regularization_strengths:
    # L1 Regularization
    model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000)
    model_l1.fit(X_train_scaled, y_train)
    train_accuracies_l1.append(model_l1.score(X_train_scaled, y_train))
    test_accuracies_l1.append(model_l1.score(X_test_scaled, y_test))
    
    # L2 Regularization
    model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=C, max_iter=2000)
    model_l2.fit(X_train_scaled, y_train)
    train_accuracies_l2.append(model_l2.score(X_train_scaled, y_train))
    test_accuracies_l2.append(model_l2.score(X_test_scaled, y_test))

# Plotting with consistent lengths
plt.figure(figsize=(10, 6))
plt.plot(regularization_strengths, train_accuracies_l1, label="L1 Train Accuracy", marker='o')
plt.plot(regularization_strengths, test_accuracies_l1, label="L1 Test Accuracy", marker='x')
plt.plot(regularization_strengths, train_accuracies_l2, label="L2 Train Accuracy", marker='o')
plt.plot(regularization_strengths, test_accuracies_l2, label="L2 Test Accuracy", marker='x')
plt.xscale('log')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.title('Train/Test Accuracy vs Regularization Strength (L1 and L2)')
plt.legend()
plt.show()


# In[ ]:




