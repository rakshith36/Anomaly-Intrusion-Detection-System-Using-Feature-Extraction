#Import modules
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
from sklearn import datasets
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import statsmodels.formula.api as sm
import statsmodels.tools
# Load "student-mat.csv" as Mdf and "student-por.csv" as Pdf.
Mdf = pd.read_csv('E:\\E drive\\D Data\\ML\\asdf\\asdf\\export\\student\\student-mat.csv',sep=';')
Pdf = pd.read_csv('E:\\E drive\\D Data\\ML\\asdf\\asdf\\export\\student\\student-por.csv',sep=';')

#dimention of datasets
print('Shape of Mdf:{}'.format(Mdf.shape))
print('Shape of Pdf:{}'.format(Pdf.shape))

fig = plt.figure(figsize=(14,5))
plt.style.use('seaborn-white')
ax1 = plt.subplot(121)
plt.hist([Mdf['G1'], Mdf['G2'], Mdf['G3']], label=['G1', 'G2', 'G3'], color=['#48D1CC', '#FF7F50', '#778899' ], alpha=0.8)
plt.legend(fontsize=14)
plt.xlabel('Grade', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.title('Math Grades', fontsize=20)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.ylim(0,220)

ax2 = plt.subplot(122)
plt.hist([Pdf['G1'], Pdf['G2'], Pdf['G3']], label=['G1', 'G2', 'G3'], color=['#48D1CC', '#FF7F50', '#778899' ], alpha=0.8)
plt.legend(fontsize=14)
plt.xlabel('Grade', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.title('Portuguese Grades', fontsize=20)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.ylim(0,220)

#plt.show()


#Add column "Subject" that describes which course the student has taken.
Mdf['Subject'] = 'M'
Pdf['Subject'] = 'P'
#Identify students who took both Math and Portuguese classes.
df = pd.merge(Mdf,Pdf,on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"], suffixes=('_M','_P'))
#Update "Subject" with "B", meaning students took both courses.
df['Subject'] = 'B'

print(df.shape)
print(df)
print(*df.columns, sep='\n')
#plot of G3, Portuguese over Math
fig = plt.figure(figsize=(7,5))
ax = sns.regplot(df.G3_M, df.G3_P, color='#778899')
ax.set(xlabel='Average Math Score', ylabel='Average Portuguese Score')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Final Score (G3), Portuguese vs Math', fontsize=16)
plt.show()

#OLS regression of final Portuguese score over final Math score.
X = np.array(df.G3_M)
Y = np.array(df.G3_P)
model = sm.OLS(Y, statsmodels.tools.add_constant(X))
results = model.fit()
print(results.summary())


#Math dataset
#create Aalc
Mdf.loc[:,'Aalc'] = (Mdf['Dalc']*5 + Mdf['Walc']*2)/7
#remove not interested variables
#"paid" has "no" values for all entries, so we will also drop it.
Mdf = Mdf.drop(['G1', 'G2', 'Dalc', 'Walc', 'paid'], axis=1)

#Repeat the same steps for Portuguese dataset
#create Aalc
Pdf.loc[:,'Aalc'] = (Pdf['Dalc']*5 + Pdf['Walc']*2)/7
#remove not interested variables
Pdf = Pdf.drop(['G1', 'G2', 'Dalc', 'Walc'], axis=1)

#Math
#Identify target variable y and predictor variables X.
ym = Mdf['G3']
Xm = Mdf[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
       'health', 'absences', 'Subject', 'Aalc']]
#Convert dummy variables values into 0/1.
Xm.school = Xm['school'].replace(['GP', 'MS'], [1,0])
Xm.sex = Xm['sex'].replace(['F','M'],[1,0])
Xm.address = Xm['address'].replace(['U','R'], [1,0])
Xm.famsize = Xm['famsize'].replace(['LE3','GT3'], [1,0])
Xm.Pstatus = Xm['Pstatus'].replace(['T','A'], [1,0])
Xm.schoolsup = Xm['schoolsup'].replace(['yes','no'],[1,0])
Xm.famsup = Xm['famsup'].replace(['yes','no'],[1,0])
Xm.activities = Xm['activities'].replace(['yes','no'],[1,0])
Xm.nursery = Xm['nursery'].replace(['yes','no'],[1,0])
Xm.higher = Xm['higher'].replace(['yes','no'],[1,0])
Xm.internet = Xm['internet'].replace(['yes','no'],[1,0])
Xm.romantic = Xm['romantic'].replace(['yes','no'],[1,0])
#Identify norminal variables
norminal_vars = ['Fjob', 'Mjob', 'Subject', 'reason','guardian']
#Convert norminal variables to dummy variables
Xm = pd.get_dummies(Xm, columns= norminal_vars, drop_first=True)
# Split data into training and test data sets.
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size = 0.3, random_state=42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
def decisiontree (X_train, y_train, X_test, y_test):
    param_grid = {'max_depth': range(1,100)}
    grid = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print ('Best cross validation score: {:.2f}'.format(grid.best_score_))
    print ('Best parameters:', grid.best_params_)
    print ('Test score:', grid.score(X_test, y_test))
    clf = RandomForestClassifier()
    print('np',np.mean(cross_val_score(clf,X_train,y_train,cv=10)))
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

decisiontree(Xm_train, ym_train, Xm_test, ym_test)