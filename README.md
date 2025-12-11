# Data_Project
# Analysis of Ideological Polarization and Disinformation on Social Networks

Team members :

The objective of this project is first to detect potential misinformation by training on the “PHEME Rumour and Stance Classification Dataset”. Then we will link this disinformation with the political party of the tweet’s writer by using our pretrained model on the “POLITiCES 2023: Analysis of political discourse on Twitter” dataset.

# 1.	Exploratory Analysis of the Dataset :
We used the PHEME Rumour and Stance Classification Dataset for training.
This dataset is made of 297 tweets and 4263 comments/reply tweets, for a total of 4560 entries. Each entry represents a tweet (text) annotated with metadata fields including event (the associated incident), threadid and tweetid (unique identifiers linking tweets to conversation threads), support (indicating whether the tweet supports, denies, or is neutral toward the claim), evidentiality (the type of evidence cited, such as a URL, quoted source, or witness account), and certainty (the annotator’s confidence level, e.g., certain, somewhat-certain, or uncertain). 

The reply tweets include responsetype-vs-source and responsetype-vs-previous describing the nature of a reply (e.g., comment, agreed, disagreed, or appeal-for-more-information). This dataset is designed for studying rumor verification, stance detection, and information credibility across social media discussions.

Concerning the metadata on disinformation, there are the columns is_rumour (rumour or non-rumour), category (the topic of the tweet), misinformation (0 if the topic was later proven true, 1 if it was misinformation), and links (a list of links/url that covered the topic).
An example of an entry :
<img width="1082" height="211" alt="image" src="https://github.com/user-attachments/assets/a18215b3-9bc5-49ac-8209-c397023eff64" />



For some entries the misinformation value was missing so we removed them because they weren’t useful for our study.

The class distribution is imbalanced here because there are a lot more true tweets than misinformation tweets.
The tweets’ lengths are mostly between 120 and 140 characters for both class (true and misinformation). 
The most common words are “Ferguson”, “Sydney”, “police” for the tweets about true stories and “War”, “Parliament”, “reports”, for the tweets spreading misinformation. (picture of the words clouds ?)

Our initial hypothesis is that misinformative tweets are related to a more extreme political view (and a more negative sentiment ?).

# 2.	Text Vector Representations :
We split the data into Train/Val/Test (70/15/15), and compared three strategies :

 •	TF-IDF (Scikit-learn) : mesure the importance of a word by how often it appear (TF = Term Frequency = how often a word appears in a document, IDF = Inverse Document Frequency = how rare the word is across all documents). The main advantage of this representation is that it is simple and fast.

 •	Word2Vec (Gensim) : words that appear in similar contexts get similar vectors. The main benefit is that it captures semantic meaning and analogies.

 •	DistilBERT (contextual embeddings, Hugging Face Transformers) : it uses a transformer model to produce contextual embeddings. So, the meaning of a word depends on the sentence around it. The advantage is that it captures both syntax and semantics but it is heavier to run.

# 3.	Models and evaluation :

We implemented and compared, with each of the three representations, the following models :

 •	Linear SVM (LinearSVC) : finds a "line" that best separates classes
 
 •	Logistic Regression (Scikit-learn) : estimates the probability that an input belongs to a class, using the logistic function. It is less heavy to run than the other models.
 
 •	MLP (PyTorch) : it is a feedforward neural network. It has nonlinear activations to learn complex patterns. It's a more flexible model, it can capture complex patterns but needs more data and tuning.

We used accuracy_score and f1_score to evaluate our models.
Results :
<img width="650" height="400" alt="image" src="https://github.com/user-attachments/assets/1563caac-e90c-4d2a-a9ac-32910ed96f72" />

The results appear to be correct, with values around 0.7-0.8 for the accuracy_score and lower values for the f1_score.

# 4.	Hyperparameter tuning and model selection :

DistilBERT fine-tuned (Hugging Face) : it is a transformer model trained on large text corpora, then fine‑tuned for a specific task, here classify misinformation. It understand the context deeply and the fine tuning adapt the model to the specific task and to the dataset. It's a more complex model that has a heavy computational cost.
 <img width="1527" height="761" alt="image" src="https://github.com/user-attachments/assets/495b6166-040b-4304-9aa9-1c45e621da17" />

 <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/930fe0db-9f83-40ec-8060-5fe723669c65" />


We can see that the results don't change much between the models but change between the reprsentations. So the text representation matter more here than the model choice.

Word2Vec is the worst representation in our case, maybe because it can't take into account word-order information like the negations (it can't tell the difference between "rumour" and "non rumour" for example).

Then, surprisingly the TF-IDF representation has better results than the BERT one and even better than the fine-tuning one. In the TF-IDF representation, words that are frequent in a tweet but rare in the general dataset are given a high weight. So, this representation captures discriminating words (hashtags, proper names, ...) well, even if it ignores context. In this dataset the tweets that support a story (category) have most of the time hashtags or names in common. So when the category is proven to be misinformation, the discriminating words are the most helpful to categorize the tweets. Looking at the stance, the propotion of replies deniying a tweet can help improve the performance. Indeed, when a tweet is misinformative the proportion of denying replies quadruples.

So, our chosen representation is **TF-IDF** and for our classification we choose the **MLP**, because it has a slightly better f1-score.

# Extension Work :

After the surprising performance of the TF-IDF representation, we thought of building a simple model classifying the tweets as misinformative if they contained too much words associated with the “misinformative lexicon”. So, we created lists of words related to misinformation, and lists of links associated to unreliable websites. Then, if the tweet contains words from those lists a penalty is added and when the total score exceeds a predetermined threshold, the tweet is classified as disinformation. This strategy will help to determined to what extend this “misinformative lexicon” is the key to identify misinformative tweets.
We obtain the following results :
<img width="945" height="469" alt="image" src="https://github.com/user-attachments/assets/d96250f7-ef7a-4deb-b1eb-1b514363b3c8" />

Those results are quite solid for a rule-based classifier. Moreover, the dataset is imbalanced and noisy, so those results demonstrate that this rule is a good way to figure out if a tweet is misinformation or not.

The model mostly classify the tweet as misinformation or not according to the reliability of the links and not as much according to the lexicon. This is also our reflex as humans, to determine if a news is trustworthy we look for the sources. So it’s a good indicator, although the results are still lower than for most of the models trained before. Supervised learning stays the best way to predict data patterns.


# Conclusions :

links between polarization and misinformation
