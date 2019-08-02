---
layout: post
title: "Twitter Political Leaning"
categories: []
tags: [machine-learning, papers, fake-news]
description: A study of quantifying and inferring the political leanings of Twitter users as a convex optimization problem to understand political demographics and temporal dynamics of users.
cover: "/assets/images/leftvsright.png"
cover_source: "https://www.kisspng.com/png-left-wing-politics-republican-party-left%E2%80%93right-p-5569097/preview.html"
comments: true
mathjax: false
---

### Introduction 

In the current times, with the widespread use of social media networks by people to voice their opinions about policies and politics in general, it has led to a big set of data that enables new avenues of research in computational political science, such as:

- answering questions political and social science:
    + proving/disproving the existence of **media bias**
    + **echo chamber** effect
- predicting election results
- personalizing social media feeds to provide a **fair and balance view of people's opinion** on controversial issues

It is not only possible now to **quantitatively study** the political demographics of users but also understand the temporal shifts in behaviour of this political polarization.

The two basic trends seen in this domain are,

- users are **consistent** in their actions of tweeting or retweeting about political issues
- similar users tend to be retweeted by **similar** audience

In context of Twitter, there are two key challenges that are to be considered,

- Is is possible to assign meaningful **numerical socres** to tweeters of their positions in the political spectrum?
- How can one devise a method that leverages the scale of Twitter data while respecting the **rate limits** imposed by the Twitter API?

The method proposed by authors uses popular Twitter users who have been retweeted many times, in particular two sets of informations associated with them,

- **Time series aspect of tweets and retweets**: User's tweet content should be consistent with who they retweet, i.e. if a user tweets a lot during a political event, they are expected to also retweet a lot at the same time.
- **Network aspect of retweeters**: Similar users get followed and retweeted by similar audience due to **homophily principle**.

The major findings of the research can be summarized as,

- Parody Twitter accounts have a **higher tendency to be liberal**. They also tend to be **temporally less stable**.
- Liberals dominate the **population of less vocal** Twitter users with less retweet activities.
- For highly vocal populations, the **liberal-conservative split is balanced**. **Partisanship also increases** with vocalness of the population.
- Hashtag usage pattern **changes rapidly** as political events unfold.
- While the event is happening, the influx of Twitter users participating makes the **active population more liberal** and less polarized.

### Related Work

- **Ideal Point Estimation**: estimating the political leaning of legislators using the roll call data and bill texts through statistical inference of their positions in an assumed latent space.
- **News Media Bias Quatification**: 
    + Indirect method: linking media outlets to reference points with known political positions by linking sentiment of headlines to economic indicators, checking co-citation of think tanks, analysing newspaper for phrases used more commonly by political members of an oriented party etc. 
    + Direct methods: analyzing news content for explicit (dis)approval of political parties and issues. These techniques also give an opportunity to track any shift in leanings of a news house over a period of time.
- **Political Polarization in Online Social Media**:
    + analysing link structure to uncover polarization of blogosphere.
    + using user-voting data to classify users and news articles
    + inferring the political orientation of news stories by the sentiment of user comments
    + assigning political leanings to search engine by linking them with political blogs
    + classifying twitter users based on linguistic content, mention/tweet behavior and social network structure
    + applying label propagation to a retweet graph for user classification

## REFERENCES:

<small>[Quantifying Political Leaning from Tweets, Retweets, and Retweeters](https://ieeexplore.ieee.org/abstract/document/7454756){:target="_blank"}</small><br>