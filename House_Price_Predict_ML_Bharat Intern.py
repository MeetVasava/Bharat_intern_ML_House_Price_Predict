#!/usr/bin/env python
# coding: utf-8

# ## NAME: - Meetkumar Vasava
# ## MACHINE LEARNING INTERNSHIP @ BHARAT INTERN

# # PROJECT NAME - **HOUSE PRICE PREDICTION**

# ## Github Link
# ## https://github.com/MeetVasava/Bharat_intern_ML_House_Price_Predict

# #  Dataset Exploration and Preprocessing

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


# Load the dataset from CSV
df = pd.read_csv('house_data.csv')


# In[4]:


# Exploratory Data Analysis (EDA)
# Let's take a quick look at the first few rows of the dataset
print(df.head())


# In[5]:


# Summary statistics of the dataset
print(df.describe())


# In[6]:


# Check for missing values
print(df.isnull().sum())


# In[7]:


# Correlation matrix to understand feature relationships
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[8]:


# Preprocessing: Selecting features and target variable
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y = df['price']


# In[9]:


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Building the Linear Regression Model

# In[10]:


# Building the Linear Regression Model
model = LinearRegression()


# In[11]:


# Fitting the model on the training data
model.fit(X_train, y_train)


# # Model Evaluation

# In[12]:


# Model Evaluation
y_pred = model.predict(X_test)


# In[13]:


# Mean Squared Error and R-squared for model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



# In[14]:


print("Mean Squared Error:", mse)
print("R-squared:", r2)


# #  Predictions and Visualization

# In[19]:


# Predictions and Visualization
# To visualize the predictions against actual prices, we'll use a scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()





# In[20]:


# We can also create a residual plot to check the model's performance
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# In[21]:


# Lastly, let's use the trained model to make predictions on new data and visualize the results
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]
predicted_price = model.predict(new_data)

print("Predicted Price:", predicted_price[0])


# In[ ]:





# In[ ]:




