import streamlit as st
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
from datetime import datetime, timedelta

st.title('Video Streaming Social Media Analyticss')
st.subheader("A social media computing project by Wan Zulmuhammad Harith, Amirah Anis Adlin, Nurlaili Hamimi and Amirul Ikhmal")

st.markdown("""## Followers Twitter Age Group""")

df = pd.read_csv('Datasets/iqiyi_followers.csv')
df['created_at'] = pd.to_datetime(df['created_at'])
df['year'] = df['created_at'].dt.strftime('%Y')
df_age = df[['screen_name', 'year']]
df_age['year'] = pd.to_numeric(df_age['year'])
today = datetime.now() - timedelta()
this_year = datetime.strftime(today, '%Y')
df_age['age'] = int(this_year) - df_age['year']
# df_age

total_count = pd.DataFrame(df_age.age.value_counts().reset_index().values, columns=["age", "aggregate age"])
a = total_count.sort_values('age')
# a

age = a['age']
count = a['aggregate age']

fig, ax = plt.subplots(figsize =(16, 9)) 

# Horizontal Bar Plot 
ax.barh(age, count) 

# Remove axes splines 
for s in ['top', 'bottom', 'left', 'right']: 
	ax.spines[s].set_visible(False) 

# Remove x, y Ticks 
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none') 

# Add padding between axes and labels 
ax.xaxis.set_tick_params(pad = 5) 
ax.yaxis.set_tick_params(pad = 10) 

# Add x, y gridlines 
ax.grid(b = True, color ='grey', 
		linestyle ='-.', linewidth = 0.5, 
		alpha = 0.2) 

# Show top values 
ax.invert_yaxis() 

# Add annotation to bars 
for i in ax.patches: 
	plt.text(i.get_width()+0.2, i.get_y()+0.5, 
			str(round((i.get_width()), 2)), 
			fontsize = 10, fontweight ='bold', 
			color ='grey') 

# Add Plot Title 
ax.set_title('Age Groups of Followers', 
			loc ='left', ) 

# Show Plot 
st.pyplot(fig)

# ---------------end of twitter age group --------------------

st.markdown("""## Friends Twitter Age Group""")

df = pd.read_csv('Datasets/iqiyi_friends.csv')
df['created_at'] = pd.to_datetime(df['created_at'])
df['year'] = df['created_at'].dt.strftime('%Y')
df_age = df[['screen_name', 'year']]
df_age['year'] = pd.to_numeric(df_age['year'])
today = datetime.now() - timedelta()
this_year = datetime.strftime(today, '%Y')
df_age['age'] = int(this_year) - df_age['year']
# df_age

total_count = pd.DataFrame(df_age.age.value_counts().reset_index().values, columns=["age", "aggregate age"])
a = total_count.sort_values('age')
# a

age = a['age']
count = a['aggregate age']

fig, ax = plt.subplots(figsize =(16, 9)) 

# Horizontal Bar Plot 
ax.barh(age, count) 

# Remove axes splines 
for s in ['top', 'bottom', 'left', 'right']: 
	ax.spines[s].set_visible(False) 

# Remove x, y Ticks 
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none') 

# Add padding between axes and labels 
ax.xaxis.set_tick_params(pad = 5) 
ax.yaxis.set_tick_params(pad = 10) 

# Add x, y gridlines 
ax.grid(b = True, color ='grey', 
		linestyle ='-.', linewidth = 0.5, 
		alpha = 0.2) 

# Show top values 
ax.invert_yaxis() 

# Add annotation to bars 
for i in ax.patches: 
	plt.text(i.get_width()+0.2, i.get_y()+0.5, 
			str(round((i.get_width()), 2)), 
			fontsize = 10, fontweight ='bold', 
			color ='grey') 

# Add Plot Title 
ax.set_title('Age Groups of Followings', 
			loc ='left', ) 

# Show Plot 
st.pyplot(fig)

# ---------------end of friends twitter age group --------------------

st.markdown("""## Followers Growth""")

followers = pd.read_csv("Datasets/iqiyi_followers_count.csv")

plt.figure(figsize=(20, 10))
plt.plot(followers['Date Collected'], followers['Count'])
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()
st.pyplot(plt)

avg_fll_per_day = followers['Count'].sum()/14
# st.write("Average followers count daily: ", int(avg_fll_per_day))

i = 0
total_fll_per_day = 0
fll_growth_per_day = 0

while i < 13:
    fll_per_day = followers['Count'][i+1] - followers['Count'][i]
    print(fll_per_day)
    i+=1
    
    total_fll_per_day = total_fll_per_day + fll_per_day

fll_growth_per_day = total_fll_per_day/14

st.write("Followers Growth Per Day: ", int(fll_growth_per_day))

# ---------------end of followers growth --------------------

st.markdown("""## Average Mentions per Day""")
df = pd.read_csv('Datasets/iqiyi_mentions.csv')

df['created_at'] = pd.to_datetime(df['created_at'])
df['date'] = [d.date() for d in df['created_at']]
df['time'] = [d.time() for d in df['created_at']]

avgtweet = df.groupby('date')['text'].count()

avgtweet = avgtweet.to_frame()
avgtweet = avgtweet.rename(columns={"text": "tweets"})

st.bar_chart(data=avgtweet, width=20)

# ---------------end of average mention per day --------------------

st.markdown("""## Peak Time of Mentions Per Day""")

mentions = pd.read_csv('Datasets/iqiyi_mentions.csv')
mentions['created_at'] = pd.to_datetime(mentions['created_at'])

bins = [0, 6, 12, 16, 20, 24]
labels = ['Midnight', 'Morning', 'Noon', 'Evening', 'Night']

mentions['time_bin'] = pd.cut(mentions['created_at'].dt.hour, bins, labels=labels, right=False)
count_mentions = mentions['time_bin'].value_counts().rename_axis('time_bin').to_frame('counts')
count_mentions.sort_values(by=['time_bin'], ascending=True, inplace=True)

count_mentions.plot(kind="bar")
plt.title("Peak Time of Mentions per Day")
st.pyplot(plt)

# ---------------end of peak time --------------------

st.markdown("""## Peak Time of Tweets Per Day""")

tweets = pd.read_csv('Datasets/iqiyi_tweets.csv')
tweets['created_at'] = pd.to_datetime(tweets['created_at'])

bins = [0, 6, 12, 16, 20, 24]
labels = ['Midnight', 'Morning', 'Noon', 'Evening', 'Night']

tweets['time_bin'] = pd.cut(tweets['created_at'].dt.hour, bins, labels=labels, right=False)

count_tweets = tweets['time_bin'].value_counts().rename_axis('time_bin').to_frame('counts')

count_tweets.sort_values(by=['time_bin'], ascending=True, inplace=True)
count_tweets.plot(kind="bar")
plt.title("Peak Time of Tweets per Day")
st.pyplot(plt)

# ---------------end of peak time --------------------

st.markdown("""## Top 10 Hashtags in Brand Mentions""")

hashtaglist = []
for obj in df['entities']:
    split = obj.split(']')
    split2= split[0].split('[')
    split3 = split2[1]
    if 'text' in split3:
        split4 = split3.split(':')
        split4 = split4[1].split(',')
        split5 = split4[0].replace('"', "")
        split5 = split5.replace("'", "")
        hashtaglist.append(split5)
        
hashtag = pd.Series(hashtaglist)
# hashtags = hashtag.value_counts().to_frame('counts')
# hashtags[:10].plot.bar()

hashtags = hashtag.value_counts()
hashtags[:10].plot(kind="bar")
plt.title("Top Hashtags")
st.pyplot(plt)