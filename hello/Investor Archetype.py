#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load dataset
df = pd.read_csv("startup_funding (1).csv")

# Drop rows with null InvestorsName or Amount
df = df[['InvestorsName', 'AmountInUSD']].dropna()

# Remove commas and convert AmountInUSD to float
df['AmountInUSD'] = df['AmountInUSD'].replace('[\$,]', '', regex=True)
df['AmountInUSD'] = pd.to_numeric(df['AmountInUSD'], errors='coerce')
df = df.dropna()

# Group by Investors
investor_profiles = df.groupby('InvestorsName').agg({
    'AmountInUSD': ['count', 'sum', 'mean', 'std']
}).reset_index()

# Flatten column names
investor_profiles.columns = ['Investor', 'DealsCount', 'TotalInvestment', 'AvgInvestment', 'StdDevInvestment']


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Fill NaN standard deviations with 0
investor_profiles['StdDevInvestment'] = investor_profiles['StdDevInvestment'].fillna(0)

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(investor_profiles[['DealsCount', 'TotalInvestment', 'AvgInvestment', 'StdDevInvestment']])

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
investor_profiles['Cluster'] = kmeans.fit_predict(scaled_data)

# Save model and scaler
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save clustered data
investor_profiles.to_csv("models/investor_clusters.csv", index=False)


# In[ ]:




