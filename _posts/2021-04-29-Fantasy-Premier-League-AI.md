---
layout: post
title: Using AI to pick a Fantasy team
subtitle: Can you use AI to pick a team for the Fantasy Premier League?
cover-img: /assets/img/fpl/cover_image_vienna_reyes.jpg
tags: [ai-series, fpl]
---

[Fantasy Premier League](https://fantasy.premierleague.com/) is one of the most widely played fantasy sports games with about 7 million players registered for the current season. The goal is simple: with a budget of £100m, pick a team with 11 players in the starting lineup and 4 on the bench and get the most points out of them. Unlike the NFL Fantasy draft system where you pick a team at the beginning of the season and aren't allowed to change it thereafter, FPL works by allowing users to change their teams week in, week out, making it more immersive. [I have played this game for more than 10 years](https://fantasy.premierleague.com/entry/260765/history) and now that I work with machine learning, I am aiming to pick a team every week with the help of AI.


## Rules

Before I explain my proposed approach, let's get into the rules for the game so we can understand the premise for building a model. They are pretty straightforward:

1. Each fantasy player starts with the same budget of £100m and has to buy 15 players (11 for the starting lineup + 4 subs).
2. You need to have 2 Goalkeepers (GK), 5 Defenders (DEF), 5 Midfielders (MID), and 3 Forwards (FWD).
3. You can have a maximum of 3 players from the same PL team.


<figure align="center">
  <img width="700" height="800" src="/assets/img/fpl/fpl_team.png" alt="Fantasy Premier League Team -  Vikas Nataraja">
    <!-- <figcaption> Data from Gameweek 1 of the 2018/19 season</figcaption> -->
</figure>
<br/>

There are a couple of other rules but these are major overarching ones. With these constraints, let's see what approaches we can take to build a model.

## How are points given out?

FPL assigns points based on goals scored, assists recorded and clean sheets. There is also a Bonus Points System (BPS) that assigns bonus points based on various other factors like crosses into the box.

- Forwards get 4 points for a goal scored, 3 points for an assist.
- Midfielders get 5 points for a goal scored, 3 points for an assist.
- Defenders get 6 points for a goal scored, 3 points for an assist, 4 points for a clean sheet, -1 for every 2 goals conceded.
- Goalkeepers get 7 points for a goal scored, 3 points for an assist, 4 points for a clean sheet, -1 for every 2 goals conceded, +1 for every 6 saves made.

In addition, if a player plays more than 60 minutes in a game, they get 2 points, otherwise just the 1. A yellow card results in a 1 point deduction, a red card means 3 points are deducted.


## What does the data look like?

FPL data is quite hard to come by but I found [this repository](https://github.com/vaastav/Fantasy-Premier-League) that does a great job of storing old information and updating it week by week. Let's look at what some of that data looks like:

<figure align="center">
  <img width="800" height="300" src="/assets/img/fpl/data_2018_19.png" alt="Fantasy Premier League Data from 2018/19">
    <figcaption> Data at the end of the 2018/19 season</figcaption>
</figure>
<br/>
How about some gameweek by gameweek data?

<figure align="center">
  <img width="800" height="300" src="/assets/img/fpl/gameweek1_data.png" alt="Fantasy Premier League Gameweek Data from 2018/19">
    <figcaption> Data from Gameweek 1 of the 2018/19 season</figcaption>
</figure>
<br/>

## Challenges

One of the most obvious challenges is that football, by its nature, is unpredictable. A 20th placed Sheffield United can beat a top 4 side like Manchester United. Form is an important factor in football and 'form over fixtures' is a popular slogan among FPL players. I've had to learn the hard way to set emotions aside when playing FPL. One of the big challenges I see in this dataset is the number of variables and the amount of variance in each of those. Just because a player performed well in Gameweeks 4, 5 and 6 doesn't mean he'll perform in Gameweeks 7 and 8. Then, there are the players who have a burst of points during limited stretches where they bring home double-digit points but then 2 points on other days. So, my model will have to learn how to predict those purple patches so I can maximize the points return.

Of course, there are certain unpredictable factors at play - injuries, players being rested for a midweek game, managerial changes, etc. My initial approach will ignore these factors as there will be too many uncertainties otherwise.

## Potential Solutions

Despite having to pick a team of 15 players, only the 11 in the starting lineup contribute towards the points total. So, that tells me that I need to prioritize choosing very good 11 players and 4 average players instead of equally good 15 players. Now, obviously, "good" is a vague definition. Often times, a "good" player won't end up getting points due to the way the game hands out points which is primarily based on goals and assists and clean sheets. This is why players like N'Golo Kante of Chelsea is not a popular FPL player despite being one of the best players in the league. FPL also has a strange way of assigning forwards as midfielders with Mo Salah, Heung Min Son and Aubameyang being some of the famous examples.

With that information, here are some ideas I have:

- Build a temporal network like RNN or LSTM that takes into account past information to predict the future. The sample size will be important here - do I prioritize the last 5 games of a player from a bottom half team or do I keep a underperforming but top player in the team? These are questions that the network will have to learn to answer.

<figure align="center">
  <img width="800" height="400" src="/assets/img/fpl/lstm.jpg" alt="LSTM architecture">
    <figcaption>Long Short Term Memory Networks. <a href="https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/">Image Source</a></figcaption>
</figure>
<br/>

- Build a decision tree or random forest as part of wider ensemble learning approach like bagging or boosting. This will allow for multiple models to work on different factors and crucially, be able to contribute individually and combine their learned features to a unified prediction. This will also help reduce the variance in the data.

<figure align="center">
  <img width="700" height="500" src="/assets/img/fpl/bagging.png" alt="Ensemble Learning - Bagging">
    <figcaption>Bagging( boostrap + aggregating) technique that uses multiple models for a prediction. <a href="https://medium.com/ml-research-lab/bagging-ensemble-meta-algorithm-for-reducing-variance-c98fffa5489f">Image Source</a></figcaption>
</figure>
<br/>

- Another way could be to use unsupervised learning to let the model figure out patterns by itself without providing labels. The bagging approach will require me adding labels to the data i.e finding the best 11 players from each gameweek, which isn't hard but would take some work. Unsupervised learning would remove that obstacle because I could feed all the cleaned data and have the model figure things out like specific clusters or observations. To do this, I could use autoencoders, K-mean clustering etc.

<figure align="center">
  <img width="700" height="350" src="/assets/img/fpl/unsupervised-learning.png" alt="Unsupervised Learning">
    <figcaption>Clustering is a type of unsupervised learning. <a href="https://www.ecloudvalley.com/mlintroduction/">Image Source</a></figcaption>
</figure>
<br/>

<figure align="center">
  <img width="700" height="300" src="/assets/img/fpl/autoencoder.png" alt="Unsupervised Learning">
    <figcaption>Autoencoders learn a compressed representation of the data <a href="https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd">Image Source</a></figcaption>
</figure>
<br/>

Either way, my initial approach will likely comprise of EDA (Exploratory Data Analysis) and this will easily take a few weeks due to the sheer amount of data - 4 seasons of data for over 500 unique players over a combined 150 gameweeks. This definitely seems like a slightly long project and my aim is have a model ready by the start of the 2021-22 season in August. Follow this blog for updates as I work on it over the summer -  I will use [this tag](https://vikasnataraja.github.io/tags#fpl) for this line of work. I'm pumped to be able to combine 2 of my favorite things - football and AI!

Cover Image Credit: [Vienna Reyes via Unsplash](https://unsplash.com/@viennachanges)
