#!/usr/bin/env python
# coding: utf-8

# In[1]:


# fundraise_model_trainer.py

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


# === Load Data ===
df = pd.read_csv("startup_funding (1).csv")

# === Basic Cleaning ===
df.fillna("", inplace=True)
df["Combined"] = (
    df["StartupName"].astype(str) + " " +
    df["IndustryVertical"].astype(str) + " " +
    df["SubVertical"].astype(str) + " " +
    df["CityLocation"].astype(str) + " " +
    df["InvestorsName"].astype(str)
)


# In[3]:


# Convert amount column to numeric
df["AmountInUSD"] = df["AmountInUSD"].replace("[^0-9]", "", regex=True)
df["AmountInUSD"] = pd.to_numeric(df["AmountInUSD"], errors="coerce").fillna(0)

# Final input = Combined text + Amount
df["FinalInput"] = df["Combined"] + " Amount " + df["AmountInUSD"].astype(str)

# Drop empty targets
df = df[df["InvestmentType"] != ""]


# In[4]:


# === Encode Target ===
le = LabelEncoder()
df["InvestmentTypeEncoded"] = le.fit_transform(df["InvestmentType"])

# === Train Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    df["FinalInput"], df["InvestmentTypeEncoded"], test_size=0.2, random_state=42
)


# In[5]:


# === TF-IDF + Classifier Pipeline ===
pipeline = make_pipeline(
    TfidfVectorizer(max_features=1000),
    MultinomialNB()
)


# In[6]:


pipeline.fit(X_train, y_train)


# In[8]:


# === Evaluation ===
y_pred = pipeline.predict(X_test)
from sklearn.utils.multiclass import unique_labels

labels_in_test = unique_labels(y_test, y_pred)
target_names_subset = le.inverse_transform(labels_in_test)

print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names_subset))


# In[9]:


# === Save Artifacts ===
joblib.dump(pipeline, "models/fundraise_model.pkl")
joblib.dump(le, "models/fundraise_label_encoder.pkl")

print("✅ Model and encoder saved successfully!")


# In[ ]:




