# Data_Project
# Analysis of Ideological Polarization and Disinformation on Social Networks

Team members :

The objective of this project is first to detect potential misinformation by training on the “PHEME Rumour and Stance Classification Dataset”. Then we will link this disinformation with the political party of the tweet’s writer by using our pretrained model on the “POLITiCES 2023: Analysis of political discourse on Twitter” dataset.

# 1.	Exploratory Analysis of the Dataset :
We used the PHEME Rumour and Stance Classification Dataset for training.
This dataset is made of 297 tweets and 4263 comments/reply tweets, for a total of 4560 entries. Each entry represents a tweet (text) annotated with metadata fields including event (the associated incident), threadid and tweetid (unique identifiers linking tweets to conversation threads), support (indicating whether the tweet supports, denies, or is neutral toward the claim), evidentiality (the type of evidence cited, such as a URL, quoted source, or witness account), and certainty (the annotator’s confidence level, e.g., certain, somewhat-certain, or uncertain). 

The reply tweets include responsetype-vs-source and responsetype-vs-previous describing the nature of a reply (e.g., comment, agreed, disagreed, or appeal-for-more-information). This dataset is designed for studying rumor verification, stance detection, and information credibility across social media discussions.

Concerning the metadata on disinformation, there are the columns is_rumour (rumour or non-rumour), category (the topic of the tweet), misinformation (0 if the topic was later proven true, 1 if it was misinformation), and links (?).
An example of an entry :
<img width="1082" height="211" alt="image" src="https://github.com/user-attachments/assets/a18215b3-9bc5-49ac-8209-c397023eff64" />



For some entries the misinformation value was missing so we removed them because they weren’t useful for our study.

The class distribution is imbalanced here because there are a lot more true tweets than misinformation tweets.
The tweets’ lengths are mostly between 120 and 140 characters for both class (true and misinformation). 
The most common words are “Ferguson”, “Sydney”, “police” for the tweets about true stories and “War”, “Parliament”, “reports”, for the tweets spreading misinformation. (picture of the words clouds ?)

Our initial hypothesis is that misinformative tweets are related to a more extreme political view (and a more negative sentiment ?).

# 2.	Text Vector Representations :
We split the data into Train/Val/Test (70/15/15), and compared three strategies :

 •	TF-IDF (Scikit-learn)

 •	Word2Vec (Gensim)

 •	DistilBERT (contextual embeddings, Hugging Face Transformers)

# 3.	Models and evaluation :

We implemented and compared, with each of the three representations, the following models :

 •	Linear SVM (LinearSVC)
 
 •	Logistic Regression (Scikit-learn)
 
 •	MLP (PyTorch)

We used accuracy_score and f1_score to evaluate our models.
Results :
<img width="650" height="400" alt="image" src="https://github.com/user-attachments/assets/1563caac-e90c-4d2a-a9ac-32910ed96f72" />

The results appear to be correct, with values around 0.7-0.8 for the accuracy_score and lower values for the f1_score.

# 4.	Hyperparameter tuning and model selection :

DistilBERT fine-tuned (Hugging Face)
 <img width="1527" height="761" alt="image" src="https://github.com/user-attachments/assets/495b6166-040b-4304-9aa9-1c45e621da17" />

We can see that the results don't change much between the models but change between the reprsentations. So the text representation matter more here than the model choice.

Word2Vec is the worst representation in our case, maybe because it can't take into account the negations (it can't tell the difference between "rumour" and "non rumour" for example).

Then, surprisingly the TF-IDF representation has better results than the BERT one and even better than the fine-tuning one. In the TF-IDF representation, words that are frequent in a tweet but rare in the general dataset are given a high weight. So, this representation captures discriminating words (hashtags, proper names, ...) well, even if it ignores context. In this dataset the tweets that support a story (category) have most of the time hashtags or names in common. So when the category is proven to be misinformation, the discriminating words are the most helpful to categorize the tweets.

# Extension Work :

# Conclusions :
