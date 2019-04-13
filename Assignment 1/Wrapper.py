#load librariries
import pandas as pd
import numpy as np
#from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#read data
read = pd.read_csv("mood_data_clean.csv")
names = ['column', 'id', 't,''monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown' 'appCat.utilities', 'appCat.weather']
data = pd.read_csv(names=names)

array = data.values()

X = array[:,3:29]
Y = array[:,10]


#this block of code is where the wrapper method will be executed on the X,Y array.
fit = test.fit(X, Y)
model = LogisticRegression()
#RFE wrapper will return top 10 features
rfe = RFE(model, 6)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))








