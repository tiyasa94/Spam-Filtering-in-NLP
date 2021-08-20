# Spam-Filtering-in-NLP
## Table of contents
   - [Overview](#overview)
   - [Motivation](#motivation)
   - [Format](#format)
   - [Technical Aspect](#technical-aspect)
   - [Technologies Used](#technologies-used)
   - [Credits](#credits)
### Demo
![download (1)](https://user-images.githubusercontent.com/74978788/130198799-5a6d15b7-9625-4fd2-8bff-24f3d5ef44a5.png)
### Overview
This is a simple spam classifier example based on Natural Language Processing in Python.
### Motivation
In present day of modern technology, the ratio of spam reports is increasig day by day. That's why I thought it can be a good start to get my hands dirty on this problem using ML and DL techniques.  
### Format
The SMS Spam Collection (text file: smsspam) has a total of 5,574 messages among of which 4,827 are SMS legitimate messages (86.6%) and 747 are spam messages(13.4%). The files contain one message per line. Each line is composed by two columns: one with label (ham or spam) and other with the raw text. ![download (1)](https://user-images.githubusercontent.com/74978788/130245939-911201d5-d4bc-4fcd-8404-1723490f3f80.png)

### Technical Aspect
This project is divided into three parts:
1. Cleaning and preprocessing of collected data.
2. Training various machine learning and deep learning models.
3. Comparing the accuracy of different models on the test data.
### Technologies Used
The Code is written in Python 3.9. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. I have used Google Colaboratory, or "Colab" for short which allows us to write and execute Python in browser. I have used various open-source Python packages in the solution, for example:
1. [PANDAS](#pandas)
2. [RE](#re)
3. [NLTK](#re)
4. [SCIKIT-LEARN](#scikit-learn)
5. [KERAS](#keras)
6. [NUMPY](#numpy)
#### PANDAS
PANDAS is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language and made available through the pandas module.
#### RE
Regular expressions (called REs) are essentially a tiny, highly specialized programming language embedded inside Python and made available through the re module.
#### NLTK
NLTK is a leading platform for building Python programs to work with human language data and made available through the nltk module. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers, stopwords for industrial-strength NLP libraries, and an active discussion forum.
#### SCIKIT-LEARN
SCIKIT-LEARN is made available through the sklearn module which is a simple and efficient tool for predictive data analysis built on NumPy, SciPy, and matplotlib. I have used CountVectorizer, different metrics for testing accuracy and various ML models.
#### KERAS
From Tensorflow we imported Keras which is one of the most used deep learning frameworks as it makes it easier to run new experiments. Keras API is designed for human beings, which follows best practices for reducing cognitive load as it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides.
#### NUMPY
Numpy is the fundamental package for scientific computing with Python which offers powerful multi-dimensional arrays and various numerical computing tools for free.
### Credits
The corpus has been collected by Tiago Agostinho de Almeida (http://www.dt.fee.unicamp.br/~tiago) and José María Gómez Hidalgo (http://www.esp.uem.es/jmgomez). I am thankful to the webpage - http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/ 
