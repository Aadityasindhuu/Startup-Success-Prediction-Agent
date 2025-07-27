#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers datasets scikit-learn pandas torch')


# In[2]:


get_ipython().system('pip install numpy==1.24.4 pandas==1.5.3')


# In[3]:


pip install numpy==1.24.4 --only-binary :all:


# In[4]:


import sys
print(sys.version)


# In[5]:


get_ipython().system('pip install numpy==1.26.4 pandas==2.1.4')


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[7]:


df = pd.read_csv("pitch_dataset (7).csv", encoding='cp1252')  # or 'latin1' if this doesn't work
df.head()



# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("pitch_dataset (6).csv", encoding='cp1252')  # or 'latin1' if this doesn't work
df.head()


# Clean up labels if needed (e.g., strip, lowercase)
df['Label'] = df['Label'].str.strip().str.capitalize()  # "weak"->"Weak" etc.

# Map labels to numbers (optional for evaluation)
label_map = {"Weak": 0, "Neutral": 1, "Strong": 2}
df['Label_num'] = df['Label'].map(label_map)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], df['Label_num'], test_size=0.2, random_state=42, stratify=df['Label_num']
)

# Vectorize text with TF-IDF (unigrams + bigrams)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize Logistic Regression with class_weight balanced
model = LogisticRegression(max_iter=200, class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_map.keys()))


# In[19]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Inverse the label_map to get number -> name
inv_label_map = {v: k for k, v in label_map.items()}

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with actual label names
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[inv_label_map[i] for i in range(len(inv_label_map))])
disp.plot(cmap='Reds')



# In[20]:


# Show top 10 words by class
import numpy as np

class_labels = ["Weak", "Neutral", "Strong"]
for i, label in enumerate(class_labels):
    top10 = np.argsort(model.coef_[i])[-10:]
    print(f"\nTop words for class '{label}':")
    print([vectorizer.get_feature_names_out()[j] for j in top10])


# In[24]:


# 7. Evaluation
y_pred = model.predict(X_test_tfidf)
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# 8. Save the Model and Vectorizer
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✅ Model and Vectorizer saved successfully.")


# In[22]:


import joblib

# Load saved model and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example usage
sample_text = ["This pitch is exceptionally compelling and persuasive."]
sample_tfidf = vectorizer.transform(sample_text)
predicted_label = model.predict(sample_tfidf)
print("Predicted Label:", predicted_label)


# In[25]:


import joblib

# Correct paths to model and vectorizer inside the 'models' folder
model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Label decoding
inv_label_map = {0: "Weak", 1: "Neutral", 2: "Strong"}

# Sample prediction
sample_text = ["We built a scalable platform that transforms logistics with AI."]
sample_tfidf = vectorizer.transform(sample_text)
predicted_label = model.predict(sample_tfidf)
print("🧠 Prediction:", inv_label_map[predicted_label[0]])



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




