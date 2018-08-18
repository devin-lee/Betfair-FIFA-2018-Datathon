import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn
import numpy as np
import pandas as pd
import scipy.stats as st
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as TTS, RandomizedSearchCV as RSC

wc = pd.read_excel('fRefinedz.xlsx')
rst = pd.read_excel('wc_datathon_eg.xlsx')

train = wc.loc[(wc['match_id'] == 0)].drop(columns = ['match_id'])
test = wc.loc[~(wc['match_id'] == 0)]

test_ID = test['match_id']
test_t1 = test['team_1']
test_t2 = test['team_2']
c_date = rst['date']
train1 = train.shape[0]
df = pd.concat([train, test], axis = 0).reset_index().drop(columns = ['index', 'match_id'])

num = df.select_dtypes(include=['int64', 'float64'])
cat = df.select_dtypes(include=['object']).drop(columns = ['t1Outcomes']).apply(LabelEncoder().fit_transform)
df['t1Outcomes'].replace('Win', 2, inplace = True)
df['t1Outcomes'].replace('Draw', 1, inplace = True)
df['t1Outcomes'].replace('Lose', 0, inplace = True)

df = pd.concat([cat, num, df['t1Outcomes']], axis = 1).drop(columns=['t1_goals', 't2_goals'])
train = df[:train1]
test = df[train1:]

y = train['t1Outcomes']
x = train.drop(columns = ['t1Outcomes'])
xtrain, xval, ytrain, yval = TTS(x, y, test_size = 0.25, random_state = 75, stratify = y)

xgbC = xgb.XGBClassifier(objective='multi:softprob')

rXparameter = {'n_estimators': st.randint (350, 1200), 'learning_rate': st.uniform (0.01, 0.03),
                        'gamma': st.uniform (0.05, 1), 'reg_alpha':st.uniform (0.5, 2), 'reg_lambda': st.uniform (0.5, 3),
                        'max_depth': st.randint (3, 10), 'min_child_weight': st.randint (3, 10), 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}


XGBmodel = RSC(xgbC, rXparameter, scoring='neg_log_loss', cv = 10, n_iter = 40)
XGBmodel.fit(xtrain, ytrain)
ypredX = XGBmodel.predict_proba(xval)
print(XGBmodel.best_params_)
print(log_loss(yval, ypredX))

xtest= test.drop(columns=['t1Outcomes'])
sub = pd.DataFrame(XGBmodel.predict_proba(xtest))
final = pd.concat([c_date, test_ID, test_t1, test_t2, sub], axis = 1).to_excel('ProbaPredictionX.xlsx', index = False)
