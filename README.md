# Positivity and social media
Internet users often debate on why their preferred social media stands out from the rest. Objectively speaking, a good way of determining the best one is  through  the  procedure  of  sentiment  analysis. This process consists of measuring the degree of positivity of a given set of comments.

The main focus here is to be able to train a model on a Twitter data set, and predict the level of positivity for a set of data from Reddit. As a result, the best social media between these two will be decided.

The resultant models will vary in their classifying approaches to see how differently they perform. Comments from Twitter will be gathered from an existing data set to train these models. Moreover, to obtain a better insight on the performance of the models, size of the training set will be modified.


## Features
* Sentiment Analysis of Twitter and Reddit datasets
* Two different type of classifiers : SVM and Multinomial Naive Bayes
* Stemming and Lemmatisation
* Word2vec and BOW model implementation


## Tech/framework used
* [Jupiter Notebook](https://jupyter.org)
* [Python](https://www.python.org)


## Installation
### Run the Jupiter Notebooks

Run the following code in the console
```
ipython nbconvert --to python <NotebookName>.ipynb
```
or download the repository and run it in local.

### The following libraries are needed to run the jupiter notebooks.

```
pip3 install nltk
pip3 install pandas
pip3 install gensim
pip3 install seaborn
```


## Tests
**1. **Importing the dataset into the**
```python
###Import the training data from file
df_train = pd.read_csv('TrainingTwitterFinal20K.csv')
df_train.dropna(axis=0, inplace=True)
```
**2. **Cleaning the sentences**
```python
###Apply clean to the train data
df_train['comment'] = df_train['comment'].apply(clean)
```
**3. **Transforming the sentences in Word2Vec or BOW model**
```python
vectorizer = CountVectorizer()
sentenceVectors = vectorizer.fit_transform(allSentences)

tfidf = TfidfTransformer()
train = tfidf.fit_transform(sentenceVectors)
```
**4. Build the classifier**
```python
###create a Multinomial Naive Bayes Classifier, input array for X values and labels s
clf = MultinomialNB().fit(train, labels)
```
**5. **Import the test dataset**
```python
###Retrieving the test data
df_test = pd.read_csv('400PosTestReddit.csv')
```
**6. **Predict the labels of the test data with the classifier**
```python
###calculate accuracy of the classifier
predicted = clf.predict(sentenceVectors2)

np.mean(predicted == labels)

Output: 0.8325
```

## Credits


