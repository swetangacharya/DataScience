#https://towardsdatascience.com/select-features-for-machine-learning-model-with-mutual-information-534fe387d5c8/
#https://thuijskens.github.io/2017/10/07/feature-selection/
#https://machinelearningmastery.com/information-gain-and-mutual-information/

from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.datasets import load_breast_cancer as LBC
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import SelectPercentile as SP
import numpy as np

cancer = LBC()
X = cancer['data']
y = cancer['target']


mi_scores = MIC(X,y)
#print(mi_scores)

# first dataset where we don't reduce the feature
X_train_1,X_test_1,y_train,y_test = tts(X,y,random_state=0,stratify=y)

mi_score_selected_index = np.where(mi_scores >0.2)[0]
X_2 = X[:,mi_score_selected_index]
X_train_2,X_test_2,y_train,y_test = tts(X_2,y,random_state=0,stratify=y)


mi_score_selected_index = np.where(mi_scores < 0.2)[0]
X_3 = X[:,mi_score_selected_index]
X_train_3,X_test_3,y_train,y_test = tts(X_3,y,random_state=0,stratify=y)


model_1 = DTC().fit(X_train_1,y_train)
model_2 = DTC().fit(X_train_2,y_train)
model_3 = DTC().fit(X_train_3,y_train)
score_1 = model_1.score(X_test_1,y_test)
score_2 = model_2.score(X_test_2,y_test)
score_3 = model_3.score(X_test_3,y_test)
print(f"score_1:{score_1}n score_2:{score_2}n score_3:{score_3}")



selector = SP(percentile=50) # select features with top 50% MI scores
selector.fit(X,y)
X_4 = selector.transform(X)
X_train_4,X_test_4,y_train,y_test = tts(X_4,y,random_state=0,stratify=y)

model_4 = DTC().fit(X_train_4,y_train)
score_4 = model_4.score(X_test_4,y_test)
print(f"score_4:{score_4}")
