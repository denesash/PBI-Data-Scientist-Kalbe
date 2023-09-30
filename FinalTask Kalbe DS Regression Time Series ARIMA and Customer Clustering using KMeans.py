#!/usr/bin/env python
# coding: utf-8

# # DATA CLEANSING

# # Data Customer

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# load data 
df_customer = pd.read_csv('Case Study - Customer.csv', delimiter=';')
df_customer.head()


# In[3]:


df_customer.info()


# In[4]:


df_customer.duplicated().sum()


# In[5]:


df_customer.isnull().sum()


# In[6]:


df_customer[df_customer['Marital Status'].isnull()]


# In[7]:


df_customer["Marital Status"].fillna((df_customer["Marital Status"].mode()[0]), inplace=True)


# In[8]:


df_customer.isnull().sum()


# In[9]:


df_customer['Income'] = df_customer['Income'].replace('[,]','.',regex=True).astype('float')
df_customer.head()


# ## Data Product

# In[10]:


# load data 
df_product = pd.read_csv('Case Study - Product.csv', delimiter=';')
df_product.head()


# In[11]:


df_product.info()


# ## Data Store

# In[12]:


# load data 
df_store = pd.read_csv('Case Study - Store.csv', delimiter=';')
df_store.head()


# In[13]:


df_store.info()


# In[14]:


df_store['Latitude'] = df_store['Latitude'].replace('[,]','.',regex=True).astype('float')
df_store['Longitude'] = df_store['Longitude'].replace('[,]','.',regex=True).astype('float')
df_store.head()


# ## Data Transaction

# In[15]:


# load data 
df_transaction = pd.read_csv('Case Study - Transaction.csv', delimiter=';')
df_transaction.head()


# In[16]:


df_transaction.info()


# In[17]:


df_transaction.duplicated().sum()


# In[18]:


df_transaction.isnull().sum()


# In[19]:


df_transaction['Date']=pd.to_datetime(df_transaction['Date'])
df_transaction.head()


# # DATA MERGE

# In[20]:


df_merge = pd.merge(df_transaction, df_customer, on=['CustomerID'])
df_merge = pd.merge(df_merge, df_product.drop(columns=['Price']), on=('ProductID'))
df_merge = pd.merge(df_merge, df_store, on=['StoreID'])


# In[21]:


df_merge.head()


# # DATA REGRESSION

# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot

import warnings
warnings.filterwarnings( 'ignore' )


# In[23]:


df_regresi = df_merge.groupby(['Date']).agg({
      'Qty' : 'sum'
}).reset_index()


# In[24]:


df_regresi


# In[25]:


figure = px.line(df_regresi, y='Qty', x="Date", title='Jumlah Barang Terjual Dalam 1 Tahun')
figure.show()


# ## Stationary Test

# In[26]:


from statsmodels.tsa.stattools import adfuller
print("Observations of Dickey-fuller test")
dftest = adfuller(df_regresi['Qty'],autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)


# In[27]:


decomposed = seasonal_decompose(df_regresi.set_index('Date'))

plt.figure(figsize=(8,8))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')

plt.tight_layout()


# In[28]:


# Split the data into train and test sets
cut_off = round(df_regresi.shape[0]*0.8)
df_train = df_regresi[:cut_off]
df_test = df_regresi[cut_off:].reset_index(drop=True)
df_train.shape, df_test.shape


# In[29]:


df_train


# In[30]:


df_test


# In[31]:


plt.figure(figsize=(20,5))
sns.lineplot(data=df_train, x=df_train['Date'], y=df_train['Qty']);
sns.lineplot(data=df_test, x=df_test['Date'], y=df_test['Qty']);


# In[32]:


autocorrelation_plot(df_regresi['Qty']);


# In[33]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Data 'Qty' from DataFrame
data = df_train['Qty']

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot ACF
plot_acf(data, lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')

# Plot PACF
plot_pacf(data, lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.show()


# # ARIMA

# In[34]:


df_train = df_train.set_index('Date')
df_test = df_test.set_index('Date')

y = df_train['Qty']

ARIMAmodel = ARIMA(y, order = (73, 0, 2))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(df_test))

y_pred_df = y_pred.conf_int()
y_pred_df['predictions'] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = df_test.index
y_pred_out = y_pred_df['predictions']


# In[35]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[36]:


y_actual = df_test['Qty']

# Calculating the metrics
mae = mean_absolute_error(y_actual, y_pred_out)
mse = mean_squared_error(y_actual, y_pred_out)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_actual - y_pred_out) / y_actual)) * 100

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[37]:


plt.figure(figsize=(10, 6))
plt.plot(df_test.index, y_actual, label='Actual')
plt.plot(df_test.index, y_pred_out, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Qty')
plt.legend()
plt.title('Actual vs. Predicted')
plt.show()


# In[38]:


plt.figure(figsize=(20,5))
plt.plot(df_train['Qty'], label='Data Training')
plt.plot(df_test['Qty'], color='green', label='Data Testing')
plt.plot(y_pred_out, color='black', label='ARIMA Predictions')
plt.title('Penjualan dalam 1 Tahun')
plt.legend()


# In[39]:


y_actual


# In[40]:


y_pred_out


# # DATA CLUSTERING

# In[41]:


df_merge.head()


# In[42]:


# mengidentifikasi kolom yang redundadnt/corelasi tinggi
df_merge.corr()


# In[43]:


df_cluster = df_merge.groupby(['CustomerID']).agg({
    'TransactionID' : 'count',
    'Qty' : 'sum',
    'TotalAmount' : 'sum'
}).reset_index().rename(columns={
    'TransactionID' : 'CountTransaction',
    'Qty' : 'TotalQty'
})


# In[44]:


df_cluster.head()


# ## Scatterplot Qty & TotalAmount

# In[45]:


import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(data=df_cluster, x='TotalQty', y='TotalAmount')


# ## Clustering Process

# In[46]:


from sklearn.preprocessing import StandardScaler

feats = ['CountTransaction', 'TotalQty', 'TotalAmount']
X = df_cluster[feats].values
data_cluster_normalize = StandardScaler().fit_transform(X)
new_df = pd.DataFrame(data=data_cluster_normalize, columns=feats)
new_df.describe()


# ## Elbow Method

# In[47]:


# Elbow Method
from sklearn.cluster import KMeans  
inertia = []  

# Loop through different numbers of clusters
for i in range(2, 11):
  kmeans = KMeans(n_clusters=i, random_state=0)  
  kmeans.fit(data_cluster_normalize)  
  nilai_inertia = kmeans.inertia_ 
  print('iterasi ke-', i, 'dengan nilai inertia: ', nilai_inertia)  
  inertia.append(kmeans.inertia_) 


# In[48]:


plt.figure(figsize=(7, 5))  

# Plot the line plot of inertia values
sns.lineplot(x=range(2, 11), y=inertia, color='#000087', linewidth=4)

# Plot the scatter plot of inertia values
sns.scatterplot(x=range(2, 11), y=inertia, s=300, color='#800000', linestyle='--')


# In[49]:


from yellowbrick.cluster import KElbowVisualizer

# fit model
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,11), metric='distortion', timings=True, locate_elbow=True)
visualizer.fit(data_cluster_normalize)       
visualizer.show()


# ## Silhouette Score

# In[50]:


from sklearn.metrics import silhouette_score

range_n_clusters = list(range(2, 11))
print(range_n_clusters)


# In[51]:


arr_silhouette_score_euclidean = []
for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i).fit(data_cluster_normalize)
    preds = kmeans.predict(new_df)
     
    score_euclidean = silhouette_score(new_df, preds, metric='euclidean')
    arr_silhouette_score_euclidean.append(score_euclidean)


# In[52]:


fig, ax = plt.subplots(1,2,figsize=(15, 6))
sns.lineplot(x=range(2, 11), y=arr_silhouette_score_euclidean, color='#000087', linewidth = 4, ax=ax[0])
sns.scatterplot(x=range(2, 11), y=arr_silhouette_score_euclidean, s=300, color='#800000',  linestyle='--',ax=ax[0])

sns.lineplot(x=range(2, 11), y=inertia, color='#000087', linewidth = 4,ax=ax[1])
sns.scatterplot(x=range(2, 11), y=inertia, s=300, color='#800000',  linestyle='--', ax=ax[1])


# In[53]:


# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
silhouette_scores = []  

for num_clusters in range_n_clusters:
    # Initialize kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(data_cluster_normalize)
    
    cluster_labels = kmeans.labels_
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(data_cluster_normalize, cluster_labels)
    silhouette_scores.append(silhouette_avg)  
    
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

# Create a line plot of silhouette scores
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# In[54]:


from yellowbrick.cluster import KElbowVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,11), metric='silhouette', timings=True, locate_elbow=True)
visualizer.fit(data_cluster_normalize)        
visualizer.show()


# In[55]:


from sklearn.cluster import KMeans  

kmeans = KMeans(n_clusters=4, random_state=0)  
kmeans.fit(data_cluster_normalize)  
df_cluster['cluster'] = kmeans.labels_
df_cluster.head()


# In[56]:


fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(data=df_cluster, x='TotalQty', y='TotalAmount', hue='cluster')


# In[57]:


display(df_cluster.groupby('cluster').agg(['mean']))


# In[ ]:




