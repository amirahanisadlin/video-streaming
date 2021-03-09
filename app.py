import streamlit as st
from multiapp import MultiApp
from apps import netflix, hulu, iqiyi, disneyplus # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Netflix", netflix.app)
app.add_app("Hulu", hulu.app)
app.add_app("iQiyi", iqiyi.app)
app.add_app("Disney+", disneyplus.app)

# The main app
app.run()