---
layout: post
title: "Social Learning Networks"
categories: []
tags: [machine-learning, papers, sln]
description: A type of network among students, instructors and modules of learning encoding the relationships among people and learning processes.
cover: "/assets/images/social-learning-networks.jpeg"
cover_source: "https://cdn-images-1.medium.com/max/1600/1*jSKvARO6h8PcfgFKpOWp3w.jpeg"
comments: true
mathjax: true
---

### Active Recall
1. [What is a SLN?](#introduction)
2. [What do the nodes, connections and weights represent in a SLN graph?](#characteristics-of-sln)
3. [What are the types of graphs used to represent information in SLN?](#types-of-sln-graphs)


### Introduction

- An SLN is a type of **social network between student, instructors and modules of learning**. The network consists of dynamics of learning behaviour over a variety of **graphs that represent the relationships** among people and processes involved in learning


### Characteristics of SLN

- People in network **learn through interaction**.
- Consists of functionality models and **graph-theoretic models**.
- **Nodes** represent learners, learning concepts, or both learners and concepts.
- **Links** represent the connection between the nodes.
- Links can be **undirected to denote similarity** or **directed to denote flow of information**.
- **Link weights** gives magnitude to the connections extracted through some network measurements.


### Applications for SLN

- Massive increase in popularity of **MOOCs** over the past decades has led to access of unprecedented data that can be easily used in the context of SLNs. Among these data, the discussion forums are of utmost importance as they are the primary means for interaction among students or between students and instructors and help extract SLNs.
- The blended learning offered by **FLIP** that combines the elements of online and traditional instruction is also gaining popularity. In this system, watching the lectures online becomes homework and class time is used for discussions instead. 
- In the last decade, there has also been a steady rise in the popularity of Q&A Sites which have emerged with complementary search engines that allows users to enter questions in NLP format.

**1. MOOCs Discussion Forums**

- Similar to Q&A sites but have a different set of terminologies.
- Each course has a **discussion forum** (hierarchy 1). Discussion forum comprises of **threads** (hierarchy 2): the first **post** is by the creator of thread followed by **comments** (hierarchy 3), hence leading to a 3-level hierarchy.
- A user interacting with these forums has one of following options:
    + Create a thread
    + Create a post
    + Comment on an existing post
- When threads are created, each forum allows annotation of these threads with a number categories. These categories are often inconsistent as there is no effective mechanism to force learners to abide by them consistently.
- Posts and comments can also **receive up votes and down votes** from users and staff.


**2. FLIP**

- The interaction in these classes can be used to extract the SLN as the structure of posts and comments here would resemble that of MOOCs but would be different on the following accounts:
    + **Size and sparsity**: smaller number of learners and denser connectivity among them.
    + **Informational vs conversational discussions**: While MOOC discussions are more conversational in nature, SLN from FLIP would typically not include conversational discussions because students can talk informally outside the class and also because the discussions are presided over by an instructor.


**3. Q&A Sites**

- Q&A sites allow the following functions to it's users:
    + post question
    + answer question
    + comment on answer
    + up/down vote post
    + allow asker to choose the best or acceptable answer
    + for quality assurance there are points associated with receiving up or down votes.

- Major differences can be seen between SLNs from such sites and sites from educational settings:
    + **Incentive structure**: well-defined automated set of incentives to encourage participation. These may or may not be a part of conversation in MOOCs or FLIP, more so because it is not a scalable concept for these forums as participation in MOOCs and FLIP can not be pushed for assessment of student academically.
    + **Broader concept list**: information propagated in an SLN for a course will be limited to materials associated with the subject. Also each course has it's own forum where only the students enrolled can participate. Q&A sites have some focal specificity but typically one can expect the number of concepts emerging in these SLNs to be much more broader.
    + **Single learning modality**: In a Q&A site, SLN is the only means of learning, whereas in an educational setting it is one of the four modalities: lecture videos, assessment and text resources.


### Types of SLN Graphs

These are the most commonly used graphs but are not comprehensive.

**1. Undirected graph among learners**

- Nodes represent learners
- Undirected links indicate the presence or absence of some characteristics between them. Properties could be age, geographic location, education level, whether or not they have interacted etc.
- For nodes \\(i\\) and \\(j\\), one can say for example, 
    + \\(prop_k(i,j)\\) is a binary variable that is 1 iff \\(i\\) and \\(j\\) satisfy property \\(k\\) in set of properties \\(K\\),
    + \\(P \leq \|K\|\\) is a threshold constant i.e. node \\(i\\) and \\(j\\) are connected if and only if they both satisfy at least \\(P\\) criiteria specified in \\(K\\).

$$(i, j) \in G \Leftrightarrow \sum_{k \in K} prop_k(i, j) \geq P \tag{1} \label{1}$$ 


**2. Directed graph among learners**

- Used to note flow of information in the SLN.
- A directional link frok \\(i\\) to \\(j\\) represents an answer by \\(j\\) to a question posted by \\(i\\). Several restrictions can be added to this directional flow e.g. include only "best answers" given.
- It could be a multi-graph where there is more than one link from \\(i\\) to \\(j\\) since learners can ask and answer more than one question each.


**3. Undirected graph among learners and concepts**

- Nodes are used to represent both the learners as well as the concepts.
- Key concepts are extracted in a number of ways:
    + running textual analysis to find keywords in discussions
    + using syllabus specified by the instructors
- Such graphs are generally bipartite graphs between learners and concepts.
- A setting similar to \eqref{1} represents this graph, but each property \\(k\\) represent a condition on the participation of user \\(i\\) in concept \\(j\\).


**4. Directed graph among learners and concepts**

- To depict structure of interactions in more details.
- Concept nodes are used and each question or post by a learner is handled separately, which allows tow sets of links for each post: \\((i_0, j) \in G\\) for learner \\(i_0\\) who makes the initial post, and \\((j, i_l) \in G\\) for each learner \\(i_l\\) who commented on \\(j\\).
- In a forum where up/down votes are allowed, these links can be weighted to match the net votes obtained.


### Research Objectives

**1. Predictions**

- **Performance**: ability to predict performance on assessments - homework, quiz, or exam questions - a student has not taken.
- **Drop-off rate**: predict drop-off rates of a course. This could be for an individual student or for the volume of participation in the course as a whole. Metrics on interest in such a case would be the completion rate of assignments, lecture videos, or rate of involvement in discussion forums.


**2. Recommendations**

- **Courses and topics**: to help students locate courses of interests based on their interactions with the enrolled courses. This presents an opportunity to improve the learning experience by recommending new courses and redirecting to relevant discussion forums.
- **Study buddies**: to help students locate study partners over MOOCs where the number of students participating can vary in terms of engagement, interests, demographics, geography etc. 

Recommendation algorithms could focus on similarities or, more importantly dissimilarities between users. For example, a learner who actively engages on discussion forums can be paired with one who is struggling in those topics.


**3. Peer-grading**

- In MOOCs, generally the teacher-to-student ratio is very small. As a result, it would be infeasible for a teaching staff to manually grade each submission. 
- This is generally tackled by only giving away machine gradable homework or exams, such as MCQs. But this limits the variety and quality of questions that can be posed.
- A different approach would be where students score each others work. This method lacks efficacy so far, because:
    + different students have different grading quality
    + time commitment is required for grading
- Structure of SLN might help in locating quality-graders for each assignment related to a specific topic.


**4. Personalization**

- Online education poses the question of trade-off between efficacy and scale in learning. It is statistically seen that only 10% of students enrolling in a MOOC ever complete the courses.
- The ineffectiveness of MOOCs can be because of the following reasons:
    + teacher-to-student ratio is very low
    + learning is asynchronous
    + student population is very diverse and hard to personalize
- Advance technology is required for course individualization, to lift tradeoff curve and enable effective learning environment at massive scales rather than having a one-size-fits-all online course.
- The information stored  in SLNs can play a key role in such adaptation to become a part of learning experience. 

![Fig-1: Flowchart of individualizaiton](/assets/2019-06-02-social-learning-networks/fig-1-flowchart-of-individualization.png?raw=true)

- The key components of such an indivualization effort can be seen in the flowchart in Fig-1.
    + **Behavioural measurements**: measurement of user behaviour while engaging with course material, e.g. video watching trajectory (pauses and jumps) can be captured, information that user enters in a discussion forum can also be collected.
    + **Data analytics**: use machine learning techniques to generate a low-dimensional model of the high-dimensional process of learning. The latent space can be:
        * discovered throught data mining
        * defined in advance in terms of author-specified learning features
    + **Content/presentation adaptation**: based on the analysis, user's updated profile dictates decisions on what content will be presented next and how it will be presented, e.g. different versions of text and video may be presented. 


### Methodologies

**1. Data Collection**

- There are two basic modes of data collections, both with pros and cons:
    + **Use existing data**: various open online course offerings over the past years remain open even after the sessions end and gives access to the discussion forums etc. Similary the SLN data from Q&A portals is accessible through the respective websites. This data can easily be crawled and scraped by writing scripts to extract the information from the pages. Major drawbacks of this methodology are:
        * no opportunity to excite the state of SLN formation for subsequent data analysis
        * only data on open courses is available
        * public data is only accessible upto a certain measurement granularity
    + **Generate new data**: To overcome the cons of earlier method, one can collaborate with the educators. Or alternatively, a team could invest resources in creating  a brand new online education platform to host courses for a number of instructors.


**2. Analysis**

- This approach varies widely based on the research objective being tackled and generally involves large-scale machine learning methods.
- Linear regression models can be used to determine which course properties are correlated with learner participation in the forums, which can be quantified using the number of posts that appeared each day for each course in the dataset.
- Using user-quiz pair matrix, algorithms can be trained:
    + baseline predictor for solving least square optimization to minimize error in terms of student and quiz biases.
    + neighbourhood predictorthat extends the baseline to leverage student-student and quiz-quiz similarities


### Conclusions

- SLN data from the online interactions can lead to tracking of information in the network that can assist various objectives that lead to better learning outcomes.


## REFERENCES:

<small>[Social Learning Networks: A brief survey](https://ieeexplore.ieee.org/document/6814139){:target="_blank"}</small><br>