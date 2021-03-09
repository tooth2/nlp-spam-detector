# nlp-spam-detector

## Building a Spam Classifier 
Spam detection is one of the major applications of Machine Learning. Pretty much all of the major email service providers have spam detection systems built in and automatically classify such mail as 'Junk Mail'. In this task we use the Naive Bayes algorithm to create a model that can classify dataset SMS messages as spam or not spam, based on the training we give to the model. It is important to have some level of intuition as to what a spammy text message might look like.

**What are spam messages?** 
Usually they have words like 'free', 'win', 'winner', 'cash', 'prize', or similar words in them, as these texts are designed to catch the user's eye and appeal to open them. Also, spam messages tend to have words written in all capitals and also tend to use a lot of exclamation marks. To the recipient, it is usually pretty straightforward to identify a spam text and our objective here is to train a model to do that for us!
Being able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. Also, this is a supervised learning problem, as we know what are trying to predict. We will be feeding a labelled dataset into the model, that it can learn from, to make future predictions.

### Bayes Theorem 
Bayes Theorem is one of the earliest probabilistic inference algorithms. It was developed by Reverend Bayes (which he used to try and infer the existence of God no less), and still performs extremely well for certain use cases. In layman's terms, the Bayes theorem calculates the probability of an event occurring, based on certain other probabilities that are related to the event in question. It is composed of "prior probabilities" - or just "priors." These "priors" are the probabilities that we are aware of, or that are given to us. And Bayes theorem is also composed of the "posterior probabilities," or just "posteriors," which are the probabilities we are looking to compute using the "priors"

### Naive Bayes 
The term 'Naive' in Naive Bayes comes from the fact that the algorithm considers the features that it is using to make the predictions to be independent of each other, which may not always be the case. So in our example, we are considering only one feature, that is the test result. Say we added another feature, 'exercise'. Let's say this feature has a binary value of 0 and 1, where the former signifies that the individual exercises less than or equal to 2 days a week and the latter signifies that the individual exercises greater than or equal to 3 days a week. If we had to use both of these features, namely the test result and the value of the 'exercise' feature, to compute our final probabilities, Bayes' theorem would fail. Naive Bayes' is an extension of Bayes' theorem that assumes that all the features are independent of each other.

### Text Processing 
Data(text) gets processed in order to use it in models.
- tokenization, stemming, and lemmatization.
- part of speech tagging and named entity recognition.

### Project Approach 
In this mission we will be using the Naive Bayes algorithm to create a model that can classify dataset SMS messages as spam or not spam, based on the training we give to the model. It is important to have some level of intuition as to what a spammy text message might look likeBeing able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. Also, this is a supervised learning problem, as we know what are trying to predict. We will be feeding a labelled dataset into the model, that it can learn from, to make future predictions.

This project has been broken down into the following steps:
- Introduction to the Naive Bayes Theorem
- Understanding our dataset
- Data Preprocessing
- Bag of Words(BoW)
- Implementing BoW from scratch
- Implementing Bag of Words in scikit-learn
- Training and testing sets
- Applying Bag of Words processing to our dataset.
- Bayes Theorem implementation from scratch
- Naive Bayes implementation from scratch
- Naive Bayes implementation using scikit-learn
- Evaluating our model
- Conclusion

### Reference 
- [Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip)
- [Dataset Intro](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Part of Speech tagging](http://www.coli.uni-saarland.de/~thorsten/publications/Brants-ANLP00.pdf)
