---
layout: post
title: "Google Smart Reply"
categories: []
tags: [NLP, machine-learning, papers]
description: Google's smart reply, a feature available in Inbox, Gmail and Allo, saves time by suggesting quick responses to the messages. This feature already drives 12% of replies over mobile devices.
cover: "/assets/images/gmail_smart_reply.png"
cover_source: "https://cdn.vox-cdn.com/thumbor/-djnqzlq6zDPf7TVXWw86bkfpKk=/0x0:577x397/1200x800/filters:focal(243x153:335x245)/cdn.vox-cdn.com/uploads/chorus_image/image/54826065/gmail_smart_reply.0.png"
comments: true
mathjax: true
---

### Introduction
Smart reply is an end to end method for automatically generating **short yet semantically diverse** email repsonses. The feature also depends on some novel methods for **semantic clustering of user-generated content** that requires minimal amount of explicitly labeled data.

Google reveals that around 25% of the email responses are 20 tokens or less in length. The high frequency of short replies was the major motivation behind developing an automated reply assist feature. The system exploits concepts of machine learning such as fully-connected neural networks, LSTMs etch.

Major challenges that have been addressed in building this features includes the following:

* High **repsonse quality** in terms of language and content.
* **Utility** maintained by presenting a variety of responses.
* **Scalable architecture** to serve millions of emails google handles without significant latencies.
* Maintaining **privacy** by ensuring that no personal data is leaked while generating training data. **Only aggregate statistics are inspected.**




## REFERENCES:

<small>[Smart Reply: Automated Response Suggestion for Email](https://ai.google/research/pubs/pub45189){:target="_blank"}</small><br>
<small>[Save time with Smart Reply in Gmail](https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/){:target="_blank"}</small>