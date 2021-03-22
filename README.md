
# NLP Codealong


```python
from src.student_caller import one_random_student
from src.student_list import quanggang
```


```python
from sklearn.datasets import fetch_20newsgroups

# Import our best friends
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
data = fetch_20newsgroups()
data.keys()
```


```python
data.target_names
```

<a id='eda'></a>

## EDA

As always, we want to look at the basic shape of the data.  


```python
X = pd.DataFrame(data['data'])
X.shape
```


```python
X.head()
```

What form do we want the above dataframe to take? What does a row represent? What does a column represent?


```python
# your answer here
```


```python
one_random_student(quanggang)
```

Let's take a look at one record.  What type of preprocessing steps should we take to isolate tokens of high semantic value?


```python
X.iloc[3].values[0]
```

Answer here


```python
one_random_student(quanggang)
```

## Frequency Distributions

Let's look at the frequency distribution of all the words in the corpus.  To do so, we will use the FreqDist class from nltk.  

The FreqDist methods expect to receive a list of tokens, so we need to do a little preprocessing. We will use the RegexpTokenizer from nltk.  

There are a few places in this notebook where regular expressions will prove useful. 

Let's look at this tool [regexr](https://regexr.com/) and try to figure out the very basic pattern to match any word.



```python
# Instantiate a RegexpTokenizer object and pass that pattern as the pattern argument

from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer 

rt = RegexpTokenizer()
```


```python
one_random_student(quanggang)
```


```python
#__SOLUTION__
# Pass that pattern into our RegexpTokenizer

from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer 

rt = RegexpTokenizer(pattern = '\w+' )
```


```python
# Join all of the words 
all_docs = ' '.join(list(X_train[0]))

# use the rt object's tokenize method to create a list of all of the tokens
all_words = None
```


```python
one_random_student(quanggang)
```


```python
#__SOLUTION__
# Join all of the words 
all_docs = ' '.join(list(X_train[0]))
# use the rt object's tokenize method to create a list of all of the tokens
all_words = rt.tokenize(all_docs)
```


```python
# Instantiate a FreqDist object and pass allwords into it

# use the most_common method to see the 10 most common words

```


```python
one_random_student(quanggang)
```


```python
#__SOLUTION__
# Instantiate a FreqDist object and pass allwords into it
fd = FreqDist(all_words)

# use the most_common method to see the 10 most common words
fd.most_common(10)
```

## Visualize the distribution of the target with a bar chart


```python
y = data['target']
```


```python
# Target classes
data['target_names']
```


```python
# Bar Chart Here (horizontal, preferably)
```


```python
one_random_student(quanggang)
```


```python
#__SOLUTION__
fig, ax = plt.subplots(figsize=(10,8))
counts = np.unique(y, return_counts=True)[0]
labels = np.unique(y, return_counts=True)[1]
ax.barh(counts, labels)
ax.set_yticks(range(0,len(data.target_names)))
ax.set_yticklabels(data.target_names)
ax.set_title('Distribution of Target Classes')
ax.set_ylabel;
```

## Quick Model

Our model validation principles are consistent with NLP modeling.   
We split our data in the same way, ideally with a hold out set.   



```python
# Train Test Split

```


```python
#__SOLUTION__
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
X_train[0]
```

## Count Vectorizor

A count vectorizor takes as input all of the documents in their raw form.  That being the case, if we are doing any preprocessing, such as custom transformations like lemming and stemming, we will need to recombine the tokens into the original documents.  

For our FSM, we will pass our documents into the vectorizer in their raw form.


```python
from sklearn.feature_extraction.text import CountVectorizer

# instantiate a CountVectorizor object 
cv = None
```


```python
one_random_student(quanggang)
```


```python
#__SOLUTION__
from nltk.corpus import stopwords

cv = CountVectorizer(stop_words=stopwords.words('english'), token_pattern='[a-zA-Z]+' )
```


```python

```

### Question: 

Look at all those wonderful parameters.  What parameters would be useful to test out? 

Let's look at our regular expressions again, and add a better pattern.

[regexr](https://regexr.com/)


```python
# Instantiate a better CountVectorizer with stopwords, a regular expression pattern, and whatever else you would like  
```


```python
one_random_student(quanggang)
```

With our CountVectorizer, we apply the same principles of model validation as we have with other data.  Fit on the training set, and transform both the train and test with that fit object. This will create a vocabulary associated with high predictive value built off of the training vocabulary. 


```python
cv.fit_transform(X_train[0])
```

### DataFrame from sparse and get feature names

As we see above, the fit_transform method returns a sparse matrix.  Luckily, our alogrithms will handle sparse matrices, as we will see below.  But, if we want, we can convert our sparse matrix to a fully expressed dataframe using the .from_spmatrix method taken from DataFrame.sparse


```python
# convert the sparse matrix from above to a dataframe
X_train_vec = None
```


```python
one_random_student(quanggang)
```


```python
#__SOLUTION__
X_train_vec = pd.DataFrame.sparse.from_spmatrix(cv.fit_transform(X_train[0]))
```

We can also add the words as column names using cv.get_feature_names()


```python
# Add words as column names
```


```python
#__SOLUTION__
# Add words as column names
X_train_vec.columns = cv.get_feature_names()
X_train_vec.head()
```

As mentioned above, we don't necessarily need the feature names present to build our model.

Let's build a model with the count vectorizer from above, and use sklearns pipeline and cross_validate to see how accurately we can classify the documents.

We will apply a CountVectorizor and then a multinomial naive bayes classifier.


```python
# import make_pipeline
# import MultinomialNB
# import cross_validate

# create a pipeline object with our CountVectorizer and Multinomial Naive Bayes as our steps

# feed the pipeline into cross_validate along with X_train[0] and y_train
```


```python
one_random_student(quanggang)
```


```python
#__SOLUTION__ 
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate

# create a pipeline object 
fsm_pipe = make_pipeline(CountVectorizer(stop_words=stopwords.words('english'), token_pattern='[a-zA-Z]+' ), MultinomialNB() )
cross_validate(fsm_pipe, X_train[0], y_train, return_train_score=True, scoring='f1_micro')
```

Now that we have a funcitonal pipeline, we have the framework to easily test out new parameters and models. Try n-grams, min_df/max_df, tfidf vectorizers, better token patterns.  Try Random Forests, XGBoost, and SVM's. The world is your oyster.

![MrBean_oysters](https://media.giphy.com/media/KZepR2JrdDbI0NYVMs/giphy.gif)


```python
#__SOLUTION__
'Random Forest did not perform very well'
from sklearn.ensemble import RandomForestClassifier
fsm_pipe = make_pipeline(CountVectorizer(stop_words=stopwords.words('english'), 
                                         token_pattern= "[a-zA-Z]+(?:'[a-z]+)?", 
                                         min_df=3, 
                                        max_df=10), RandomForestClassifier(n_estimators=10) )

cross_validate(fsm_pipe, X_train[0], y_train, return_train_score=True)
```


```python
# Of course, when we are finished tuning our model, we fit on the entire training set, and score on the test.
fsm_pipe.fit(X_train[0], y_train)
```


```python
y_hat_test = fsm_pipe.predict(X_test[0])
```


```python
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(fsm_pipe,X_test[0], y_test)
```
