Introduction:  
Online marketplaces have grown in popularity over the last few decades, online sellers and 
merchants now ask their customers to provide feedback on the products they have purchased. 
Every day, millions of evaluations are written about various products, services, and locations 
online. This   has made the Internet the most significant of acquiring concepts and beliefs about 
a product or a service. However, as the number of reviews available for a product increases, it 
becomes more difficult for a potential consumer to make an informed decision about whether 
or not to purchase the product. Different opinions on the same product on the one hand, and 
ambiguous reviews on the other, make it more difficult for customers to make the right 
decision. Here, it appears that all e-commerce enterprises must analyze these contents. 
Review/Sentiment analysis and classification is a computational study that attempts to address 
this problem by extracting subjective information such as opinions and sentiments from natural 
language texts. Natural language processing, text analysis, computational linguistics, and 
biometrics are just a few of the methods that have been explored to approach this issue. In 
recent years, Machine learning And Deep learning methods have been well-liked in the 
semantic and review analysis for their simplicity and accuracy. 

Libraries Used:  
1. NUMPY - It is a Python library used for working with arrays. It also has functions for 
working in particularly four domains of linear algebra, fourier transform, and matrices. 
NumPy was created in 2005 by Travis Oliphant. It is an open-source project and you 
can use it freely. NumPy stands for Numerical Python. 
2. PANDAS -  It is an open-source library that is made mainly for working with relational 
or labeled data both easily and intuitively. It provides various data structures and 
operations for manipulating numerical data and time series. This library is built on top 
of the NumPy library. Pandas is fast and it has high performance & productivity for 
users. 
3. SKLEARN -  Scikit Learn or Sklearn is one of the most robust libraries for machine 
learning in Python. It is an open-source and built upon NumPy, SciPy, and Matplotlib. 
It provides a range of tools for machine learning and statistical modeling including 
dimensionality reduction, clustering, regression, and classification, through a consistent 
interface in Python. Additionally, it provides many other tools for evaluation, selection, 
model development, and data preprocessing. Scikit-learn is one of NumFOCUS’s 
fiscally sponsored projects. It also integrates well with many other Python libraries, 
such as Matplotlib, Plotly, NumPy, Pandas, SciPy, etc. Although the library is fairly 
new, it has quickly become one of the most popular libraries on GitHub. A number of 
big organizations such as Spotify, Evernote, JP Morgan, Inria, AWeber, and many more 
use Sklearn. Sklearn is used to build machine learning models. 
4. NLTK - Natural language processing (NLP) is a field that focuses on making natural 
human language usable by computer programs. NLTK, or Natural Language Toolkit, 
is a Python package that you can use for NLP. A lot of the data that you could be 
analyzing is unstructured data and contains human-readable text. Before you can 
analyze that data programmatically, you first need to preprocess it. In this tutorial, 
you’ll take your first look at the kinds of text preprocessing tasks you can do with NLTK 
so that you’ll be ready to apply them in future projects. You’ll also see how to do some 
basic text analysis and create visualizations. 
5. TEXTBLOB - TextBlob is a Python library for processing textual data. It provides a 
simple API for common natural language processing (NLP) tasks such as part-of
speech tagging, noun phrase extraction, sentiment analysis, classification, translation, 
and more. TextBlob is built on top of NLTK (Natural Language Toolkit) and Pattern, 
and it provides a simple and intuitive API that makes it easy to use for text processing 
tasks in Python. 
6. WORDCLOUD - A word cloud is a visual representation of text data, where the size of 
each word indicates its frequency or importance within the text. Words that appear more 
frequently are displayed with larger fonts or in a more prominent position in the cloud. 
Word clouds are often used to quickly visualize the most prominent terms in a given 
text or dataset. Creating a word cloud involves feeding textual data into a word cloud 
generator, which then calculates word frequencies and arranges the words in a visually 
appealing manner. It is a useful starting point for exploring and summarizing the 
content of textual data quickly. 
7. MATPLOTLIB - It is an excellent Python visualization library for 2D array charts. 
Matplotlib is a cross-platform data visualization toolkit built on NumPy arrays and 
designed to work with the entire SciPy stack. John Hunter first mentioned it in 2002. 
One of the most important advantages of visualization is that it provides us with visual 
access to massive amounts of data in easily 19 understandable images. Line, bar, scatter, 
histogram, and other plot types are available in Matplotlib. 
8. TENSORFLOW - It is a robust open-source machine learning framework, enables 
developers and academics to use cutting-edge machine learning from start to finish. Its 
extensive, scalable ecosystem of tools, libraries, and community resources enables 
rapid development and deployment of ML applications.

Dataset Collection - Nykaa Review dataset from Kaggle is being used here. This 
dataset contains 278 677 reviews. From this dataset, a total of 55000 reviews have been 
taken, which is used for this project. This dataset has been labeled with three classes, 
which are positive, negative, and neutral. 

Data Preprocessing –  
▪ Punctuation Remove: The removal of punctuation from textual data is the most 
common text processing technique. The process of removing punctuation will aid 
in treating each text equally. For example, after the punctuation is removed, the 
words data and data! are treated equally. We must also exercise extreme caution 
when selecting the list of punctuations to exclude from the data based on the use 
cases. Python's string. punctuation contains these symbols! "#$%&\'()*+,
./:;?@[\\]^ {|}~` 
▪ Removing Stopwords: Stopwords are words that are commonly used to describe 
less meaningful concepts, such as article, pronoun, preposition, and so on. Stop 
using NLP and text mining. In general, words are removed. Different stop terms are 
used in various formats depending on the country, language, and other factors. 
There may be many meaningless words in a document. If we do not remove these 
words, the classifier will have a difficult time determining the correct result. 
▪ Tokenization: Tokenization is the process of dividing long strings of input text into 
smaller chunks. It converts a string or document to tokens (smaller chunks). 
Tokenization is most commonly used to divide values such as a document, sentence, 
or paragraph into smaller units such as words or subwords. These smaller units are 
referred to as tokens. It is a step in the process of preparing a text for natural 
language processing. 
▪ Word Embedding: Word embedding produces a numerical representation of textual 
data. It offers equivalent representations for words with similar semantic features 
to help in word differentiation.

CLASSIFIERS: 
This section gives a brief overview of the algorithms used in this project report. Here, is the 
evaluation of six machine learning models in our system. MNB, LR, DT, RF, SVM and KNN 
are the six algorithms under machine learning. 
❖ Multinomial Naive Bayes (MNB) - When categorizing texts based on a statistical 
examination of their contents, the multinomial naive Bayes algorithm is frequently utilized. 
It offers an alternative to "heavy" AI-based semantic analysis and significantly streamlines 
the classification of textual material. The classification aims to assign text fragments (i.e. 
documents) to classes by determining the likelihood that a document belongs to the same 
class as other documents with the same subject. 
❖ Support Vector Classifier (SVC) - A Linear SVC (Support Vector Classifier) is used to 
fit the data you provide, returning a "best fit" hyperplane that divides or categorizes your 
data. After you have obtained the hyperplane, you can feed some features into your 
classifier to see what the "predicted" class is. The Linear SVC model has more parameters 
than the SVC model, such as penalty normalization (L1 or L2) and loss function. 
❖ Logistic Regression (LR) - The most basic version of logistic regression, though there are 
many more complex variations, employs a logistic function to describe a binary dependent 
variable. Regression analysis employs the logistic regression method (also known as logit 
regression) to estimate the parameters of a logistic model (a form of binary regression). 
❖ Decision Tree (DT) - The Decision Tree algorithm is a member of the supervised learning 
algorithm family. The goal of using a decision tree is to build a training model that can be 
used to predict the class or value of the target variable by learning simple choice rules 
generated from prior data (training data). Decision trees are a type of supervised machine 
learning that divides data indefinitely based on a parameter. Using the binary tree from 
earlier, one can comprehend a decision tree. Most decision tree algorithms work top-down, 
selecting a variable that best divides the set of objects at each stage. 
❖ Random Forest (RF) - During the training phase of the random forests or random decision 
forests ensemble learning approach, which is used for classification, regression, and other 
tasks, a large number of decision trees are built. For classification problems, the random 
forest output is the class that the majority of the trees chose. For regression tasks, the mean 
or average prediction of each tree is returned. In RF, the classifier data variables are drawn 
at random from a large number of trees. 
❖ K-Nearest Neighbors (KNN) - This algorithm can be used for both regression and 
classification tasks. K-Nearest Neighbors examines the labels of a predetermined number 
of data points surrounding a target data point to predict which class the data point belongs 
to. K-Nearest Neighbors (KNN) is a conceptually simple yet extremely powerful algorithm, 
and it is one of the most widely used machine learning algorithms.
