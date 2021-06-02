---
layout: post
title: An Analysis of the 2020/21 Fantasy Premier League Season
subtitle: Data-driven insights about the 2020/21 Premier League season
cover-img: /assets/img/fpl_analysis/cover_image_nathan_rogers.jpg
tags: [ai-series, fpl]
---

After a strange, intense eight months and 11 days filled with twists and turns, the 2020/21 Premier League season has come to a close. And with that, the Fantasy Premier League has concluded for all 8.24 million managers as well. In this blog post, I'll take a look at some of the most important takeaways from the fantasy season. We will analyze points, cost, most valuable players, price changes, what worked, what didn't and much more. The player graphics used for the plots in this post come from the [official FPL website](https://fantasy.premierleague.com/). The data comes from their API. The code I wrote to analyze the data and generate these plots is publicly available [here](https://github.com/vikasnataraja/fantasy-premier-league). This post is part of a [series exploring](https://vikasnataraja.github.io/tags#fpl) the use of data and AI in Fantasy Premier League.

## Most Valuable Players

I am terming this section `Most Valuable Players` taking inspiration from the NBA because they gave us the best bang for the buck. That means these are not necessarily the players who simply accumulated the most points over the season but rather how many points they produced for their value. To analyze this, I first looked at the player-by-player-gameweek-by-gameweek data. In total, according to the official FPL website, 713 players were registered across the 20 teams at the start of the season. Among them, only 262 scored more than 50 points!

To characterize "value", I use the points they produced for every $£ 1m$ spent by a manager. Therefore, if a player cost more, they would have to return a high total points score to be termed "Most Valuable". For instance, among the forwards, although Harry Kane accumulated an extraordinary 242 pts across the season, he only ranks 10th for value because his average cost over the season was a whopping $£ 12.6m$. That is still a remarkable return. No one symbolized Leeds's season better than **Patrick Bamford** who tops the charts, producing 194 pts at an average cost of just $£7.1m$. Calvert-Lewin's form dipped in and out at certain points but comes in for a cool 4th place finish. Callum Wilson and McGoldrick battled relegation but put up decent points returns.

<figure align="center">
  <img width="800" height="400" src="/assets/img/fpl_analysis/best_value_fwds.png" alt="Fantasy Premier League Forward MVPs by Vikas Nataraja">
    <!-- <figcaption> Most Valuable Forwards of the 2020/21 season</figcaption> -->
</figure>
<br/>

Among midfielders, every valuable player in the top 10 cost less than $£ 6m$ on average. And again, Manchester United's Bruno Fernandes put up incredible numbers but is not in the top 10 as he had a very high price. The MVP among midfielders is West Brom's Matheus Pereira by quite a margin, costing only $£ 5.1m$ but producing an excellent 153 pts. Gundogan's goalscoring heroics ensure that he comes in 2nd while Leeds are again well represented with both Raphinha and Harrison having good seasons.

<figure align="center">
  <img width="800" height="400" src="/assets/img/fpl_analysis/best_value_mid.png" alt="Fantasy Premier League Midfield MVPs by Vikas Nataraja">
    <!-- <figcaption> Most Valuable Midfielders of the 2020/21 season</figcaption> -->
</figure>
<br/>

For the last couple of seasons, players like Trent Alexander-Arnold and Robertson have proved good value with their goals and assists but Liverpool's poor season and their high cost means they are nowhere near the top 10 for MVPs. Who does top the list is **Stuart Dallas** who brought in 171 pts. And to think, he only cost $£4.5m$ at the start of the season (but more on that later). West Ham and Aston Villa shored up their defence this season and are represented by Coufal, Cresswell, Konsa and Targett in the top 10. John Stones's impressive resurgence lands him the 10th spot.

<figure align="center">
  <img width="800" height="400" src="/assets/img/fpl_analysis/best_value_def.png" alt="Fantasy Premier League Defender MVPs by Vikas Nataraja">
    <!-- <figcaption> Most Valuable Defenders of the 2020/21 season</figcaption> -->
</figure>
<br/>

There could only be one goalkeeper who could top this list and that is, of course, **Emi Martinez**. He cost only $£5.7m$ but consistently kept clean sheets, made numerous saves and secured bonus points. A distant second place is secured by Meslier while Sam Johnstone's efforts land him the 3rd spot.

<figure align="center">
  <img width="800" height="400" src="/assets/img/fpl_analysis/best_value_gks.png" alt="Fantasy Premier League Goalkeeper MVPs by Vikas Nataraja">
    <!-- <figcaption> Most Valuable Goalkeepers of the 2020/21 season</figcaption> -->
</figure>
<br/>

## Most Bonus Points

Bonus points are a good way to make small differentials in mini-leagues. Those add up at the end of the season and could be the difference between you and your mate. FPL calculates bonus points based on a range of metrics and awards them to the 3 best players in a match. The BPS is detailed [here](https://fantasy.premierleague.com/help/rules).

So, who got the most bonus points? No surprises here as the top 2 highest scoring players of the season **Bruno Fernandes and Harry Kane** stand head and shoulders above their competition. In fact, about 14% of their total points came from just bonus points! Bamford and Martinez had great seasons and appear on this list. Trent Alexander-Arnold's late season form puts him 5th on the list. Luke Shaw is the only other defender on the list.

<figure align="center">
  <img width="800" height="400" src="/assets/img/fpl_analysis/most_bonus_pts.png" alt="Fantasy Premier League Bonus Points by Vikas Nataraja">
    <!-- <figcaption> Most Bonus Points of the 2020/21 season</figcaption> -->
</figure>
<br/>


## Popularity and Performance

Often in FPL, popularity and performance don't go hand in hand. It is one of the many reasons that makes the game unpredictable. For example, Pierre-Emerick Aubameyang was owned by more than 35% of the managers in gameweek 2 but did not deliver and by gameweek 38, he was only owned by 7%. Players go through bad patches of form while others hit purple patches and it is important to recognize both.

**Heung-Min Son** was the most owned player on average throughout the season with nearly 48% of managers selecting him in their teams. But, he only comes in 4th in terms of points returns. All the usual big-hitters - **Kane, Fernandes and Salah** - had excellent seasons once again sealing their spots in the top 4. Jack Grealish had an excellent first half of the season with nearly 40% ownership in gameweek 23 which dropped significantly after his injury. He comes in 10th.

<figure align="center">
  <img width="800" height="400" src="/assets/img/fpl_analysis/popularity_vs_performance.png" alt="Fantasy Premier League Popularity vs Performance Chart by Vikas Nataraja">
    <!-- <figcaption> Popularity and Performance of Players for the 2020/21 season</figcaption> -->
</figure>
<br/>


## Biggest Price Swings

Finally, let's see who had the best and worst seasons in terms of price changes. FPL changes the price based on performances (and selection?) meaning players who perform well get more expensive and vice-versa. Exemplifying Leeds's season are Bamford and Dallas, both rewarded with more than $£1m$ increase on their base price. But, nobody comes close to **Harry Kane**'s $\Delta$, starting the season at $£10.5m$ but ending it at $£11.9m$. Elsewhere, the biggest casualties of the price swing were El-Ghazi and Neal Maupay. Aubameyang is easily the biggest name on here, suffering a $£0.7m$ decrease from $£12.0m$ at the start of the season.

<figure align="center">
  <img width="800" height="400" src="/assets/img/fpl_analysis/price_swings.png" alt="Fantasy Premier League Price Swing Chart by Vikas Nataraja">
</figure>
<br/>


That's it for now, I'll be back with some more stats and analyses later. FPL data is huge and there is a lot of potential for interesting work. I am currently working on developing some insights for my own team as well as a few other select teams that will eventually end up in an AI model. The new season starts on August 14, 2021, and I am aiming to work hard this summer, so keep an eye on this space!



Cover Image Credit: [Nathan Rogers](https://unsplash.com/@nathanjayrog) via Unsplash
