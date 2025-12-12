# Data_Project  
## Analysis of Ideological Polarization and Disinformation on Social Networks

**Team members:** *Inés Menchero Garcia, Patricia Lidia Sanchez Rodriguez, Lucas Charbonnier, Ambre Bissiriex.*

The goal of this project is twofold:

1. **Detect potential misinformation** by training supervised models on the *PHEME Rumour and Stance Classification Dataset*.  
2. **Relate misinformation to ideological polarization**, using additional data on stance and network structure to study how false claims spread and how they are received by different communities.

---

## 1. Exploratory Analysis of the Dataset

We use the **PHEME Rumour and Stance Classification Dataset** as our main source of annotated data. It consists of:

- **297 source tweets**  
- **4,263 reply tweets**  
- **4,560 tweets in total**

Each row corresponds to a tweet (either source or reply) and includes:

- **Text fields**: the tweet text itself.  
- **Conversation structure**:  
  - `event`: the real-world incident the tweet refers to  
  - `threadid`, `tweetid`: identifiers linking tweets into conversation threads  
- **Stance-related annotations**:  
  - `support`: whether the tweet *supports*, *denies*, *queries*, or is *neutral* towards the main claim  
  - `evidentiality`: whether evidence is provided (URL, quoted source, reported speech, etc.)  
  - `certainty`: annotators’ perceived certainty (e.g., *certain*, *somewhat-certain*, *uncertain*)  

For reply tweets, we also have:

- `responsetype-vs-source` and `responsetype-vs-previous`, describing the nature of the reply (e.g., *comment*, *agree*, *disagree*, *appeal-for-more-information*).

These annotations make the dataset suitable for **rumour verification, stance detection and credibility analysis**.

On top of this, we focus on metadata related to (mis)information:

- `is_rumour`: distinguishes rumour vs. non-rumour  
- `category`: topic of the story (e.g., *Ferguson*, *Sydney siege*, …)  
- `misinformation`:  
  - `0` if the story was later verified as **true**  
  - `1` if the story turned out to be **misinformation**  
- `links`: external URLs that reported the story, often pointing to news outlets or blogs

An example of an entry:

<img width="1082" height="211" alt="image" src="https://github.com/user-attachments/assets/a18215b3-9bc5-49ac-8209-c397023eff64" />

For some events the `misinformation` label is missing. Since we cannot use unlabeled instances for supervised learning, we **remove these rows** from our experiments.

### Class distribution and basic statistics

- The **class distribution is imbalanced**: there are many more *true* stories than *misinformation* stories. This makes misinformative tweets more likely to be misclassified as true.
- Tweet lengths are mostly between **120 and 140 characters** for both classes (true vs. misinformation).
- A quick lexical analysis shows that:
  - In **true stories**, the most frequent keywords include *“Ferguson”*, *“Sydney”*, *“police”*.
  - In **misinformative stories**, typical words include *“war”*, *“parliament”*, *“reports”*, etc.  
  These patterns are consistent with the different events covered in the dataset.

*(Here we could include word clouds to illustrate the most frequent tokens in each class.)*

### Initial hypothesis

Our starting hypothesis is that misinformative tweets:

- Tend to be associated with **more extreme political views** and **more negative sentiment**.
- Are embedded in **echo chambers** where users preferentially consume information consistent with their prior beliefs.
- May **reinforce and amplify pre-existing ideological divisions** between communities.

The rest of the project is structured to test these ideas, focusing first on **text-based detection of misinformation**, and then on how this relates to **stance and polarization**.

---

## 2. Text Vector Representations

We split the data into **train / validation / test** sets with a **70 / 15 / 15** ratio and compare three main strategies for encoding tweet text:

1. **TF–IDF (Scikit-learn)**  
   TF–IDF measures the importance of a word based on:
   - **TF (Term Frequency)**: how often the word appears in a tweet.
   - **IDF (Inverse Document Frequency)**: how rare the word is across all tweets.
   
   This yields **sparse, high-dimensional vectors**, but the method is:
   - Simple and fast to compute.
   - Very effective at highlighting **discriminative tokens** (hashtags, names, rare keywords).

2. **Word2Vec (Gensim)**  
   Word2Vec learns **dense vector embeddings** where words that appear in similar contexts obtain similar vectors. The main advantages are:
   - Captures **semantic similarity** and analogies.
   - Produces compact numeric representations that can be averaged to represent full tweets.

   However, simple averaging of word embeddings loses **word order** and subtle compositional effects, which can be problematic for tasks where negation or local context matter.

3. **BERT-based contextual embeddings (Hugging Face Transformers)**  
   We also use **contextual embeddings** derived from a pre-trained Transformer (e.g., BERT/DistilBERT) with **frozen weights**:
   - Each tweet is encoded into a dense vector that depends on the surrounding context.
   - This representation captures both **syntax and semantics**, going beyond bag-of-words.

   The trade-off is that this approach is **computationally heavier**, but it typically yields more informative features than TF–IDF or Word2Vec in many NLP tasks.

---

## 3. Models and Evaluation

For each of the three text representations (TF–IDF, Word2Vec, BERT embeddings), we train and evaluate the following models:

- **Linear SVM (LinearSVC, Scikit-learn)**  
  A linear Support Vector Machine finds a hyperplane that maximally separates the two classes in feature space. It is:
  - Well suited to **high-dimensional sparse features** like TF–IDF.
  - Robust and usually a strong baseline for text classification.

- **Logistic Regression (Scikit-learn)**  
  Logistic Regression estimates the probability that an input belongs to the positive class using the logistic function. It is:
  - Simple, fast and easy to interpret.
  - A strong **linear baseline** that often performs surprisingly well with good features.

- **MLP (PyTorch)**  
  A Multi-Layer Perceptron is a feed-forward neural network with:
  - One hidden layer with non-linear activations, allowing it to learn **non-linear decision boundaries**.
  - More flexibility to capture complex patterns than linear models, at the cost of:
    - Higher computational cost.
    - Greater risk of overfitting if not properly regularized.

We use **accuracy** and **macro F1-score** to evaluate our models. Macro F1 gives equal weight to both classes and is therefore more informative than accuracy under class imbalance.

### Results

The final test results for all model–representation combinations are:

| Representation        | Model                 | Test Accuracy | Test F1 |
|-----------------------|-----------------------|---------------|---------|
| **BERT embeddings**   | LinearSVC             | 0.796         | 0.598   |
|                       | LogisticRegression    | 0.796         | 0.598   |
|                       | MLP (PyTorch)         | 0.817         | 0.617   |
| **Raw text (tokens)** | DistilBERT fine-tuned | 0.817         | 0.617   |
| **TF–IDF**            | LinearSVC             | 0.834         | 0.671   |
|                       | LogisticRegression    | 0.834         | 0.671   |
|                       | **MLP (PyTorch)**     | **0.846**     | **0.705** |
| **Word2Vec**          | LinearSVC             | 0.583         | 0.452   |
|                       | LogisticRegression    | 0.689         | 0.519   |
|                       | MLP (PyTorch)         | 0.817         | 0.617   |

Overall, the results are consistent and highlight clear differences between
representations. With **TF–IDF** features, all three models (Logistic
Regression, LinearSVC and MLP) perform very well, with the best configuration
being **TF–IDF + MLP**, which reaches a **test accuracy of 0.846** and a
**macro F1 of 0.705**. The linear models on TF–IDF (Logistic Regression and
LinearSVC) also obtain strong scores of **0.834** accuracy and **0.671** macro F1.

The **DistilBERT fine-tuned** model on raw text achieves a competitive
**0.817** test accuracy and **0.617** macro F1, matching the performance of
the best BERT-based and Word2Vec-based MLPs, but still below the top TF–IDF
setup. Frozen **BERT embeddings** combined with classical models reach around
**0.796** accuracy and **0.598** macro F1, while **Word2Vec** is generally the
weakest representation for linear models: its LinearSVC and Logistic
Regression variants stay in the **0.58–0.69** accuracy range with F1 around
**0.45–0.52**. Only when combined with an MLP does Word2Vec approach the
performance of the contextual and transformer-based models.

---

## 4. Hyperparameter Tuning and Model Selection

In addition to the classical models above, we fine-tune a **DistilBERT** model (Hugging Face Transformers):

- DistilBERT is a **lighter Transformer** pretrained on large text corpora.
- We fine-tune it directly on our dataset for **binary classification (misinformation vs. true)**.
- This end-to-end model:
  - Captures deep contextual information.
  - Adapts its weights to the specific properties of our tweets.
  - Comes with a **higher computational cost** than the linear baselines.


### Representation vs. model: what matters more?

The experiments show that:

- **Performance differences between models (SVM, Logistic Regression, MLP)** are relatively small **when the representation is fixed**.
- In contrast, **changing the representation** (TF–IDF vs. Word2Vec vs. BERT vs. raw text with DistilBERT) has a **much larger impact** on performance.

In particular:

- **Word2Vec** is clearly the **weakest representation for linear models**.  
  Averaging word embeddings over the tweet discards **word order** and subtle
  cues such as negations (e.g., “rumour” vs. “non-rumour”), which are crucial
  to this task. With an MLP, Word2Vec becomes competitive, but it does not
  surpass the best TF–IDF configurations.

- **TF–IDF** achieves the best overall performance in our setting.  
  With TF–IDF, all three models (Logistic Regression, LinearSVC and MLP)
  achieve strong results, and the **TF–IDF + MLP** combination clearly
  dominates with **0.846** accuracy and **0.705** macro F1. This is noticeably
  higher than the scores obtained with DistilBERT fine-tuned and with frozen
  BERT embeddings. The results show that, for this dataset, a relatively simple
  lexical representation can be extremely effective when combined with a
  well-regularised classifier.

- **BERT embeddings and DistilBERT fine-tuned** still achieve strong results.  
  The fine-tuned DistilBERT model is among the top-performing systems, very
  close to the best BERT/Word2Vec MLPs, but it does not clearly outperform the
  strongest TF–IDF models. This confirms that transformer-based architectures
  are powerful general-purpose text classifiers, but in this specific dataset
  they do not provide a decisive advantage over simpler, bag-of-words-based
  representations.

Based on these observations, our final choice for the main classifier is:

- **Representation:** TF–IDF  
- **Model:** MLP (it yields the best macro F1 and accuracy while remaining relatively efficient).

---

## Extension Work: Rule-based Misinformation Detection

Motivated by the strong performance of TF–IDF, we explored a **very simple rule-based classifier**, inspired by the idea of a “misinformation lexicon”:

1. We manually constructed:
   - A list of words and expressions frequently associated with misinformation.
   - A list of domains and URLs corresponding to **unreliable** or low-credibility sources.

2. For each tweet, we computed a **score** based on:
   - The presence of lexicon items in the text.
   - The presence of **unreliable links** in the URL list.

3. If the score exceeded a fixed threshold, the tweet was classified as **misinformation**; otherwise as **true**.

We obtain the following results:

<img width="1527" height="761" alt="image" src="https://github.com/user-attachments/assets/495b6166-040b-4304-9aa9-1c45e621da17" />

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/930fe0db-9f83-40ec-8060-5fe723669c65" />


Those results are quite solid for a rule-based classifier, especially considering that:

- The dataset is noisy and class-imbalanced.
- The classifier is based only on a hand-crafted lexicon and link heuristics.

The model mostly classifies tweets as misinformation or not according to the
**reliability of the links**, and less according to the lexicon itself. This is
similar to our own behaviour as humans: to decide whether a piece of news is
trustworthy, we often **look at the sources first**. The indicator is useful,
although the results are still lower than for most of the supervised models
trained before. This confirms that **supervised learning remains the best way
to capture complex patterns** in the data.

---

## Conclusions

Our main findings can be summarised as follows:

1. **Representation is more crucial than the choice of classifier.**  
   For this dataset, the difference between TF–IDF, Word2Vec and BERT
   embeddings (or raw text for DistilBERT) is much larger than the difference
   between Logistic Regression, Linear SVM and MLP when using the same
   representation. TF–IDF, despite its simplicity, turns out to be the most
   effective representation.

2. **Class imbalance and label noise limit performance.**  
   The dataset contains many more true than misinformative tweets. As a result,
   models often misclassify misinformation as true, and even good accuracy can
   hide relatively low recall for the minority class.

3. **Text-only features are not sufficient for robust misinformation detection.**  
   Incorporating additional signals, such as:
   - **Stance distributions** within a thread (proportion of replies denying vs. supporting the claim),
   - **Source reliability** (domains of linked URLs),
   clearly improves our understanding of how misinformation behaves.

   For example, threads linked to misinformation tend to show a **higher
   proportion of denying replies**, and the same misleading claims recur within
   particular lexical and topical clusters.

4. **Misinformation and ideological polarization are closely related.**  
   Even when a tweet is misinformative, the **majority of replies often support
   the claim**, which supports our initial hypothesis: users tend to seek and
   amplify information that confirms their existing beliefs. This contributes
   to:
   - Stronger **echo chambers**.
   - Increased **polarisation between communities**.

   The excellent performance of TF–IDF models (especially compared to more
   complex transformers) suggests that highly characteristic lexical markers
   are often reused within categories. This indicates that misinformation
   frequently circulates inside relatively homogeneous, polarized communities
   where linguistic behaviour is also more uniform.

In summary, the project shows that **simple, well-regularised models with
carefully chosen representations** can compete with much more complex
architectures, and that studying both **content** and **interaction patterns**
is essential for understanding the link between **disinformation** and
**ideological polarization** on social networks.
