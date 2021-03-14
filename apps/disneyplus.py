import streamlit as st
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
import networkx as nx
from operator import itemgetter
import community
from datetime import datetime, timedelta
from PIL import Image

def app():
	@st.cache(allow_output_mutation=True)
	def get_followers():
		df = pd.read_csv('Datasets/disneyplus_followers.csv')
		return df

	@st.cache(allow_output_mutation=True)
	def get_friends():
		df = pd.read_csv('Datasets/disneyplus_friends.csv')
		return df

	@st.cache(allow_output_mutation=True)
	def get_mentions():
		df = pd.read_csv('Datasets/disneyplus_mentions.csv')
		return df

	@st.cache(allow_output_mutation=True)
	def get_count():
		df = pd.read_csv('Datasets/disneyplus_followers_count.csv')
		return df

	@st.cache(allow_output_mutation=True)
	def get_tweets():
		df = pd.read_csv('Datasets/disneyplus_tweets.csv')
		return df

	@st.cache(allow_output_mutation=True)
	def get_nodes():
		with open('Datasets/disneyplus_nodelist.csv', 'r') as nodecsv:  # Open the file
			nodereader = csv.reader(nodecsv)  # Read the csv
			# Python list comprehension and list slicing to remove the header row)
			nodes = [n for n in nodereader][1:]
		return nodes

	@st.cache(allow_output_mutation=True)
	def get_edges():
		with open('Datasets/disneyplus_edgelist.csv', 'r') as edgecsv:  # Open the file
			edgereader = csv.reader(edgecsv)  # Read the csv
			edges = [tuple(e) for e in edgereader][1:]  # Retrieve the data
		return edges

	st.title('Video Streaming Social Media Analytics')
	st.subheader("A social media computing project by Wan Zulmuhammad Harith, Amirah Anis Adlin, Nurlaili Hamimi and Amirul Ikhmal")

	st.markdown("""## Followers Twitter Age Group""")

	df = get_followers()
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
	a = a.drop(a.loc[a['age'] >= 15].index)
	a = a.reset_index(drop=True)

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

	df = get_friends()
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

	followers = get_count()

	plt.figure(figsize=(20, 10))
	plt.plot(followers['Date Collected'], followers['Count'])
	plt.xlabel('Date')
	plt.ylabel('Count')
	st.pyplot(plt)

	avg_fll_per_day = followers['Count'].sum()/14
	# st.write("Average followers count daily: ", int(avg_fll_per_day))

	i = 0
	total_fll_per_day = 0
	fll_growth_per_day = 0

	while i < 13:
		fll_per_day = followers['Count'][i+1] - followers['Count'][i]
		i+=1
		
		total_fll_per_day = total_fll_per_day + fll_per_day

	fll_growth_per_day = total_fll_per_day/14

	st.write("Followers Growth Per Day: ", int(fll_growth_per_day))

	# ---------------end of followers growth --------------------

	st.markdown("""## Average Mentions per Day""")
	df = get_mentions()

	df['created_at'] = pd.to_datetime(df['created_at'])
	df['date'] = [d.date() for d in df['created_at']]
	df['time'] = [d.time() for d in df['created_at']]

	avgtweet = df.groupby('date')['text'].count()

	avgtweet = avgtweet.to_frame()
	avgtweet = avgtweet.rename(columns={"text": "tweets"})

	st.bar_chart(data=avgtweet, width=20)

	# ---------------end of average mention per day --------------------

	st.markdown("""## Peak Time of Mentions Per Day""")

	mentions = get_mentions()
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

	tweets = get_tweets()
	tweets['created_at'] = pd.to_datetime(tweets['created_at'])

	bins = [0, 6, 12, 16, 20, 24]
	labels = ['Midnight', 'Morning', 'Noon', 'Evening', 'Night']

	tweets['time_bin'] = pd.cut(tweets['created_at'].dt.hour, bins, labels=labels, right=False)

	count_tweets = tweets['time_bin'].value_counts().rename_axis('time_bin').to_frame('counts')

	count_tweets.sort_values(by=['time_bin'], ascending=True, inplace=True)
	count_tweets.plot(kind="bar")
	plt.title("Peak Time of Tweets per Day")
	st.pyplot(plt)

	# col1, col2 = st.beta_columns((1, 1))
	# with col1: 
	# 	st.pyplot(count_mentions)
	# with col2: 
	# 	st.pyplot(count_tweets)

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

	# ---------------end of hashtaglist --------------------

	st.markdown("""## Centrality Graph""")
	
	image = Image.open('disneyplus_graph.png')
	img_array = np.array(image)
	st.image(img_array)
	
	st.markdown("""## Centrality Measures""")
	
	st.markdown("""### Top 5 Betweenness of DisneyPlus""")
	st.text("Name: disneyplus | Betweenness Centrality: 0.03278973024928096 | Degree: 180")
	st.text("Name: MarvelStudios | Betweenness Centrality: 0.0022595071218984254 | Degree: 36")
	st.text("Name: Marvel | Betweenness Centrality: 0.002086193495320754 | Degree: 35")
	st.text("Name: Disney | Betweenness Centrality: 0.001820831345012364 | Degree: 26")
	st.text("Name: milygirl10 | Betweenness Centrality: 0.001019574881730995 | Degree: 39")
		   
	st.markdown("""### Top 5 Eigenvector of DisneyPlus""")
	st.text("Name: disneyplus | Eigenvector Centrality: 0.4818729458776124 | Degree: 180")
	st.text("Name: milygirl10 | Eigenvector Centrality: 0.19314162103905416 | Degree: 39")
	st.text("Name: LoLo68985017 | Eigenvector Centrality: 0.17880977078333604 | Degree: 32")
	st.text("Name: MarvelStudios | Eigenvector Centrality: 0.1733427072741722 | Degree: 36")
	st.text("Name: Marvel | Eigenvector Centrality: 0.16422465543850892 | Degree: 35")

	st.markdown("""### Top 5 Centrality of DisneyPlus""")
	st.text("Name: disneyplus | Degree Centrality: 180 | Degree: 180")
	st.text("Name: milygirl10 | Degree Centrality: 39 | Degree: 39")
	st.text("Name: MarvelStudios | Degree Centrality: 36 | Degree: 36")
	st.text("Name: Marvel | Degree Centrality: 35 | Degree: 35")
	st.text("Name: LoLo68985017 | Degree Centrality: 32 | Degree: 32")
		    
	
	# with open('Datasets/disneyplus_nodelist.csv', 'r') as nodecsv:  # Open the file
	# 	nodereader = csv.reader(nodecsv)  # Read the csv
	# 	# Python list comprehension and list slicing to remove the header row)
	# 	nodes = [n for n in nodereader][1:]
	# # print(nodes)
	
	# node_names = [n[0] for n in nodes]  # Get a list of only the node names
	# # print(node_names)
	
	# with open('Datasets/disneyplus_edgelist.csv', 'r') as edgecsv:  # Open the file
	# 	edgereader = csv.reader(edgecsv)  # Read the csv
	# 	edges = [tuple(e) for e in edgereader][1:]  # Retrieve the data

	# 	nodes = get_nodes()
	# 	node_names = [n[0] for n in nodes]  # Get a list of only the node names
	# 	edges = get_edges()

	# 	# create graph object
	# 	G = nx.Graph()

	# 	G.add_nodes_from(node_names)
	# 	G.add_edges_from(edges)

	# 	person_dict = dict(G.degree(G.nodes()))
	# 	nx.set_node_attributes(G, name='person_dict', values=person_dict)
	# 	# # person_dict

	# 	part = community.best_partition(G)
	# 	remove = [node for node, degree in dict(G.degree()).items() if degree < 1]
	# 	G.remove_nodes_from(remove)
	# 	plt.figure(figsize=(50, 50))
	# 	nx.draw_networkx(G, pos=nx.spring_layout(G))  # try other layouts - search networkx help for options
	# 	st.pyplot(plt)
	
	

	
