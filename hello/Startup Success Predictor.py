#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


# 1. Load Dataset
df = pd.read_csv("startup data.csv")


# In[3]:


# 2. Select Important Features
features = [
    "funding_rounds", "funding_total_usd", "relationships", "avg_participants",
    "is_CA", "is_NY", "is_MA", "is_TX", "is_otherstate",
    "is_software", "is_web", "is_mobile", "is_enterprise", "is_advertising",
    "is_gamesvideo", "is_ecommerce", "is_biotech", "is_consulting", "is_othercategory",
    "has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC", "has_roundD"
]

df = df[features + ["status"]].dropna()


# In[5]:


# 3. Encode Target
df["status"] = df["status"].apply(lambda x: 1 if x == "success" else 0)

# 4. Split Data
X = df[features]
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# 5. Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


# 6. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# In[9]:


# 7. Predict and Evaluate
y_pred = model.predict(X_test_scaled)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report
from sklearn.metrics import classification_report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))


# In[10]:


import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.title("🚀 Feature Importance - Startup Success Predictor")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


# In[11]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Average CV Score:", scores.mean())


# In[13]:


print(df.head())


# In[14]:


import pickle

# Save model inside the 'models' directory
with open("models/success_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the model column names used during training
with open("models/model_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)



# In[ ]:




