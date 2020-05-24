from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import sys
import requests
import nltk
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import re
import nltk
import math
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn import linear_model, datasets
from sklearn.metrics.pairwise import cosine_similarity
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from random import randrange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve, RandomizedSearchCV
import re
import scipy
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pandas as pd
import pandas
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from python_scrape import get_all_tweets

cache = []
soup = BeautifulSoup
nltk.download('stopwords')

def run_test_cases(random):
    print(random.predict(["It's outrageous that the Republican legislative leaders and the conservative majority on the Supreme Court in Wisconsin are willing to risk the health and safety of many thousands of Wisconsin voters tomorrow for their own political gain."]))
    print(random.predict(["Frontline workers—including grocery and pharmacy workers, warehouse workers, medical personnel, farmworkers, food processing workers, truck drivers, postal workers, delivery drivers, and janitors—must be provided hazard pay, child care, and safe working conditions."]))
    print(random.predict(["Their workers should not have to put their lives at risk to earn their paycheck. Walmart must end their greed, guarantee sick leave to all and stop putting workers at risk."]))
    print(random.predict(["At a time when millions are losing their jobs AND their health care, the American people are now seeing the gross deficiencies in our employer-based private health care system. Health care is a human right, not an employee benefit."]))
    print(random.predict(["This is a crisis. People don't know how they're going to afford to stay in their homes. We need to suspend rent and mortgage payments, evictions, and foreclosures nationwide."]))
    print(random.predict(["During this growing economic crisis, tens of millions of Americans are on the verge of going hungry. We cannot allow that to happen. The federal government must act now to make sure that every American has access to the food they need during this horrific pandemic."]))
    print(random.predict(["The cruelty and absurdity of our for-profit health care system is more obvious in the midst of this crisis than it has ever been."]))
    print(random.predict(["We cannot continue to have a dysfunctional, greedy health care system which ties most people's health care coverage to employment, leaves tens of millions uninsured and bankrupts over 500,000 people a year."]))


    print(random.predict(["CNN is giving a platform not just to Iranian propagandists, but to Iranian propagandists pushing disinformation trying to weaken sanctions against nuclear weapons and terrorism. "]))
    print(random.predict(["China should be banned from any trade deals and any support from anyone until they shut there disgusting markets and help the world recover with out continuously telling lies."]))
    print(random.predict(["Its not useless.  We have a major illegal immigration issue.  We need strong borders.  Keeping the virus out has never been the purpose of the wall.  Of course you’re a foreign national so this is none of your business."]))
    print(random.predict(["First, we don't even know how much illegal aliens are putting into the pot (estimates range from $11-$18B) because we don't know how many illegal aliens there are in the US. Secondly, illegal immigration costs about $116B a year. Whatever they put in, it doesn't break even."]))
    print(random.predict(["baby boomers are probably the worst generation in human civilization lol they left their children with a 20 trillion dollar debt and a climate change crisis that's destroying the planet"]))

def convert_text():

    data_url = '/Users/yhailu/Downloads/ExtractedTweets.csv'
    tweets = pd.read_csv(data_url)
    tweets.drop(['Handle'], axis=1, inplace=True)
    le = preprocessing.LabelEncoder()
    balance_data = tweets.apply(le.fit_transform)
    processed_features = []

    primary_url = '/Users/yhailu/Desktop/primary_debates_cleaned 3.csv'
    primary_text = pd.read_csv(primary_url)
    primary_text.columns = primary_text.columns.str.replace('Text', 'Tweet')
    primary_text.drop(['Line', 'Speaker',
                           'Date', 'Location', 'URL'], axis=1, inplace=True)

    new_df = pd.concat([tweets, primary_text], ignore_index=True)

    features = new_df.iloc[:, 1].values
    labels = new_df.iloc[:, 0].values

    for i in range(len(labels)):
        if 'Democrat' in labels[i]:
            labels[i] = 'Democrat'
        if 'Republican' in labels[i]:
            labels[i] = 'Republican'

    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        processed_features.append(processed_feature)



    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=40)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words="english")),
        ('clf', LinearSVC())
    ])

    params = {
        'vectorizer__max_df': (0.5, 0.75),
        'vectorizer__ngram_range': [(1, 3), (2,2), (2,3), (3,3)],
        'vectorizer__min_df': [5],
        'clf__max_iter': [5000],
        'clf__C': [1.0]
    }

    kfold = KFold(n_splits=10, random_state=None)
    results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
    print(results)
    random = RandomizedSearchCV(pipeline, params)
    vect = pipeline.named_steps['vectorizer']
    X = vect.fit(X_train, y_train)
    random.fit(X_train, y_train)
    rand_predictions = random.predict(X_test)

    # print classification report
    #print(classification_report(y_test, rand_predictions))

    #run_test_cases(random)


def correlation():
    pearson_df = {'Name': ['Donald Trump', 'Bernie Sanders', 'Elizabeth Warren', 'Mitch Mcconnel', 'Ted Cruz', 'AOC'],
            'Text': ['In light of the attack from the Invisible Enemy, as well as the need to protect the jobs of our GREAT American Citizens, I will be signing an Executive Order to temporarily suspend immigration into the United States!',
                     'If you live paycheck to paycheck and you lack paid sick and medical leave, staying home is not an option even when you are feeling sick. That is unfair to the employee and to people they may come in contact with. We must ensure paid leave for every worker.',
                     'Immigrants are on the front lines of the coronavirus response, putting their lives at risk to make sure our communities are fed, healthy, and safe. The Trump administration needs to stop using this pandemic as cover to implement their xenophobic agenda.',
                     'This morning, Iran’s master terrorist is dead. The architect and chief engineer for the world’s most active state sponsor of terrorism has been removed from the battlefield at the hand of the United States military.',
                     'We should NOT be releasing violent criminals. “The records also show many inmates released had multiple charges—143 people.... Charges range from drug possession to aggravated assault, aggravated robbery, assaulting a peace officer, arson, human sex trafficking....”',
                     'Now is the time to create millions of good jobs building out the infrastructure and clean energy necessary to save our planet for future generations.']
            }


    #test = 'Health care is a universal human right. Employer based health care is a scam, we need to make sure every man, woman and child in this country has quality healthcare'


    # df = pd.DataFrame(pearson_df, columns=['Name', 'Text'])
    # pearson_features = df.iloc[:, 1].values
    # pearson_labels = df.iloc[:, 0].values
    # #X_train, X_test, y_train, y_test = train_test_split(pearson_features, pearson_labels, test_size=0.2, random_state=40)
    # vectorizer = CountVectorizer()
    # feats = vectorizer.fit_transform(pearson_df)
    #
    # for each in feats:
    #     print("each", each)
    #
    # print ("DF", df)


    # test_2 = 'Medicare for all is essential for every single american. Health care for all is something that should have been granted to every man, woman and child'
    #
    # test_3 = 'We need to protect the second amendment for everyone. Gun rights are an american right'
    # test_4 = 'Guns are the most important things in their lives, they will protect it with everything they know'

    test_dt = get_all_tweets('realdonaldtrump')
    test_bens = get_all_tweets('benshapiro')
    test_mm = get_all_tweets('senatemajldr')
    test_lg = get_all_tweets('LindseyGrahamSC')
    test_ew = get_all_tweets('ewarren')
    test_bs = get_all_tweets('BernieSanders')
    test_bens = get_all_tweets('benshapiro')
    test_KK = get_all_tweets('KyleKulinski')
    test_JB = get_all_tweets('JoeBiden')

    with open('/Users/yhailu/Library/Mobile Documents/com~apple~TextEdit/Documents/test_article.txt', 'r') as myfile:
        data = myfile.read()
    print(data)

    #documents = (test, test_2, test_3)
    documents = (test_dt, test_KK, test_lg, data)
    tfidf = TfidfVectorizer(use_idf=True,  ngram_range=(2, 4))
    tfidf_matrix = tfidf.fit_transform(documents)
    # tfidf_test = tfidf.fit_transform([data])

    #iterate through each of the strings that we have provided and print the cosine similarity
    length = len(documents)
    cosine_list = []
    for i in range(length):
        cosine_list.append(cosine_similarity(tfidf_matrix[i],tfidf_matrix[len(documents) - 1]))

    print(cosine_list)


correlation()
#convert_text()


