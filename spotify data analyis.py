#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_tracks=pd.read_csv('C:\\Users\\Ishan Sharma\\OneDrive\\Desktop/tracks.csv')
df_tracks.head()


# In[ ]:


#checking null values
pd.isnull(df_tracks).sum()
    


# In[ ]:


#information of the dataset
df_tracks.info()


# In[ ]:



sorted_df= df_tracks.sort_values('popularity', ascending= True).head(10)
sorted_df


# In[ ]:


#total info of the table and transpose for better understanding
df_tracks.describe().transpose()


# In[ ]:


# top 10 most popular songs on spotify
most_popular= df_tracks.query('popularity>90', inplace= False).sort_values('popularity', ascending= False)
most_popular[:10]


# In[ ]:


df_tracks.set_index('release_date', inplace = True)
df_tracks.index=pd.to_datatime(df_tracks.index)
df_tracks.head()


# In[ ]:


#using specific data
df_tracks[['artists']].iloc[18]


# In[ ]:


# Convert duration in ms to sec
df_tracks['duration'] = df_tracks['duration_ms'].apply(lambda x:round(x/1000))
#Drop the duration_ms column
df_tracks.drop('duration_ms', inplace=True, axis=1)


# In[ ]:


# Display all the columns names
df_tracks.columns


# In[ ]:


corr_df = df_tracks.drop(["key", "mode", "explicit"], axis=1).corr(method="pearson")


# In[ ]:


#Find the correlation map
plt.figure(figsize=(14,6))
heatmap = sns.heatmap(corr_df, annot=True, fmt=".1g", vmin=-1,vmax=1, center=0,cmap="inferno",lw=0.75, linecolor="Black")
heatmap.set_title("Correlation heatmap between variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)


# In[ ]:


sample_df = df_tracks.sample(int(0.004*len(df_tracks)))
sample_df


# In[ ]:


# regression line
plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y="loudness", x="energy", color="c").set(title="Loudness vs Energy Correlation")


# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y="popularity", x="acousticness", color="r").set(title="Popularity vs Acousticness Correlation")


# In[ ]:


# Create a new column called 'dates' and get the year data in a variable called 'years'
df_tracks['dates'] = df_tracks.index.get_level_values('release_date')
df_tracks.head()
df_tracks.dates = pd.to_datetime(df_tracks.dates)
years = df_tracks.dates.dt.year


# See the years values
years.head()


# In[ ]:


sns.displot(years, discrete=True, aspect=2, height=5, kind="hist").set(title="Number of songs per year")

