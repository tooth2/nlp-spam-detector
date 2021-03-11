# nlp-spam-detector

## Building a Spam Classifier 
Spam detection is one of the major applications of Machine Learning. Pretty much all of the major email service providers have spam detection systems built in and automatically classify such mail as 'Junk Mail'. In this task we use the Naive Bayes algorithm to create a model that can classify dataset SMS messages as spam or not spam, based on the training we give to the model. It is important to have some level of intuition as to what a spammy text message might look like.

*What are spam messages?* 
Usually they have words like 'free', 'win', 'winner', 'cash', 'prize', or similar words in them, as these texts are designed to catch the user's eye and appeal to open them. Also, spam messages tend to have words written in all capitals and also tend to use a lot of exclamation marks. To the recipient, it is usually pretty straightforward to identify a spam text and our objective here is to train a model to do that for us!
Being able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. Also, this is a supervised learning problem, as we know what are trying to predict. We will be feeding a labelled dataset into the model, that it can learn from, to make future predictions.

![spam](images/dqnb.png)

### Project Approach 
In this mission we will be using the Naive Bayes algorithm to create a model that can classify dataset SMS messages as spam or not spam, based on the training we give to the model. It is important to have some level of intuition as to what a spammy text message might look likeBeing able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. Also, this is a supervised learning problem, as we know what are trying to predict. We will be feeding a labelled dataset into the model, that it can learn from, to make future predictions.

This project has been broken down into the following steps:
1. Data Preprocessing
2. Training and testing sets
3. Applying Bag of Words(BoW) processing to the dataset
4. Bayes Theorem , Naive Bayes implementation from scratch
5. Naive Bayes implementation using scikit-learn
6. Evaluating the model

### Bayes Theorem 
Bayes Theorem is one of the earliest probabilistic inference algorithms. It was developed by Reverend Bayes (which he used to try and infer the existence of God no less), and still performs extremely well for certain use cases. In layman's terms, the Bayes theorem calculates the probability of an event occurring, based on certain other probabilities that are related to the event in question. It is composed of "prior probabilities" - or just "priors." These "priors" are the probabilities that we are aware of, or that are given to us. And Bayes theorem is also composed of the "posterior probabilities," or just "posteriors," which are the probabilities we are looking to compute using the "priors"
<img src ="images/bayes_formula.png" width="200" />

### Naive Bayes 
The term 'Naive' in Naive Bayes comes from the fact that the algorithm considers the features that it is using to make the predictions to be independent of each other, which may not always be the case. So in our example, we are considering only one feature, that is the test result. Say we added another feature, 'exercise'. Let's say this feature has a binary value of 0 and 1, where the former signifies that the individual exercises less than or equal to 2 days a week and the latter signifies that the individual exercises greater than or equal to 3 days a week. If we had to use both of these features, namely the test result and the value of the 'exercise' feature, to compute our final probabilities, Bayes' theorem would fail. Naive Bayes' is an extension of Bayes' theorem that assumes that all the features are independent of each other.
<img src ="images/naivebayes.png" width="400" />

### Text Processing 
Data(text) gets processed in order to use it in models.
- Convert all strings to their lower case form.
- Removing all punctuation
- Tokenization, stemming, and lemmatization.
- Part of speech tagging and named entity recognition.

### Bag of Words
What we have here in our data set is a large collection of text data (5,572 rows of data). Most ML algorithms rely on numerical data to be fed into them as input, and email/sms messages are usually text heavy. Here we'd like to introduce the Bag of Words (BoW) concept which is a term used to specify the problems that have a 'bag of words' or a collection of text data that needs to be worked with. The basic idea of BoW is to take a piece of text and count the frequency of the words in that text. It is important to note that the BoW concept treats each word individually and the order in which the words occur does not matter.

Our objective here is to convert this set of texts to a frequency distribution matrix, as follows:
Here as we can see, the documents are numbered in the rows, and each word is a column name, with the corresponding value being the frequency of that word in the document.Let's break this down and see how we can do this conversion using a small set of documents.To handle this, we will be using sklearn's count vectorizer method which does the following:

- tokenizes the string (separates the string into individual words) and gives an integer ID to each token
- counts frequencies : the occurrence of each of those tokens
- ```from sklearn.feature_extraction.text import CountVectorizer```

![CountVectorizer](images/countvectorizer.png)

### Training/Testing Dataset
- X_train : training data for the 'sms_message' column
- y_train : training data for the 'label' column
- X_test : testing data for the 'sms_message' column
- y_test : testing data for the 'label' column

### Naive Bayesian Implementation
sklearn has several Naive Bayes implementations that we can use. We will be using sklearn's sklearn.naive_bayes method to make predictions on our SMS messages dataset.
Specifically, we will be using the **multinomial Naive Bayes algorithm**. This particular classifier is suitable for classification with discrete features (such as in our case, word counts for text classification). It takes in integer word counts as its input. On the other hand, Gaussian Naive Bayes is better suited for continuous data as it assumes that the input data has a Gaussian (normal) distribution.

### Advantage of Naive Bayes Algorithm
One of the major advantages that Naive Bayes has over other classification algorithms is its ability to handle an extremely large number of features. In our case, each word is treated as a feature and there are thousands of different words. Also, it performs well even with the presence of irrelevant features and is relatively unaffected by them. The other major advantage it has is its relative simplicity. Naive Bayes' works well right out of the box and tuning its parameters is rarely ever necessary, except usually in cases where the distribution of the data is known. It rarely ever overfits the data. Another important advantage is that its model training and prediction times are very fast for the amount of data it can handle. 

### Evaluation
- Accuracy:  measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).
- Precision : measures what proportion of messages we classified as spam, actually were spam. It is a ratio of true positives (words classified as spam, and which actually are spam) to all positives (all words classified as spam, regardless of whether that was the correct classification).  
 > True Positives/(True Positives + False Positives)
- Recall (sensitivity): measures what proportion of messages that actually were spam were classified by us as spam. It is a ratio of true positives (words classified as spam, and which actually are spam) to all the words that were actually spam. 
> True Positives/(True Positives + False Negatives)

### Reference 
- [SMSSPAM Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip)
- [Dataset Intro](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Compressed Data](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/)
- [Part of Speech tagging](http://www.coli.uni-saarland.de/~thorsten/publications/Brants-ANLP00.pdf)
- [UCI ML datasets](https://archive.ics.uci.edu/ml/datasets/News+Aggregator)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)
- [Quotes API](http://quotes.rest/)
- [Regular Expression in Python](https://docs.python.org/3/library/re.html)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [NLTK Tockenizer](http://www.nltk.org/api/nltk.tokenize.html)
