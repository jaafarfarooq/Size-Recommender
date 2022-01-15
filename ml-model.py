import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy import stats
import pickle

maledf=pd.read_csv('D:/uni stuff/FYP_Material/ANSUR_II_MALE_Public.csv' , encoding='latin-1')
femaledf=pd.read_csv('D:/uni stuff/FYP_Material/ANSUR_II_FEMALE_Public.csv' , encoding='latin-1')
maledf = maledf[[
'chestcircumference',
'shouldercircumference',
'Gender',
'Age',
'Heightin',
'Weightlbs']]
femaledf = femaledf[[
'chestcircumference',
'shouldercircumference',
'Gender',                     
'Age',
'Heightin',
'Weightlbs']]
df=pd.concat([maledf,femaledf])
df.rename(columns={
'chestcircumference':'chest',
'shouldercircumference':'shoulder',
'Heightin':'Height',
'Weightlbs':'Weight'}, inplace=True)
df.loc[(df.Gender == 'Male'),'Gender']='0'
df.loc[(df.Gender == 'Female'),'Gender']='1'
df["Gender"] = pd.to_numeric(df["Gender"])
#REMOVING OUTLIERS ON THE BASIS OF INTER QUARTILE RANGE
q1 = df.quantile(0.25)
q2 = df.quantile(0.5)
q3 = df.quantile(0.75)
iqr = q3-q1

df = df[~((df < (q1 - 2 * iqr)) | (df > (q3 + 2 * iqr))).any(axis=1)]

#REMOVING OUTLIERS ON THE BASIS OF Z-SCORE 
z = np.abs(stats.zscore(df))
df = df[(z < 3).all(axis=1)]
from sklearn.model_selection import train_test_split
X = np.asarray(df[['Age', 'Weight', 'Height','Gender']])
y = np.asarray(df[['chest','shoulder']])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn import metrics
from sklearn.linear_model import LinearRegression
ml_model=LinearRegression()
model=ml_model.fit(X_train,y_train)
pickle.dump(model, open('iri.pkl', 'wb'))
y_prediction=model.predict(X_test)
r2_score=metrics.r2_score(y_test,y_prediction)
import math 

age=int(input("Enter your age:"))
weight=float(input("Enter your weight in pounds:"))
height=float(input("Enter your height in inches:"))

gender=int(input("Enter your gender(0=male,1=female):"))
new_input = [[age,weight,height,gender]]
new_output = model.predict(new_input)
chestsize=math.ceil(new_output[0][0]/25.4)
shouldersize=math.ceil(new_output[0][1]/25.4)
if (chestsize<32):
    if(shouldersize<40):
        print("Size: Extra Small")
    else:
        print("Size: Small")
elif (chestsize>32 and chestsize<36):
    if(shouldersize<45):
        print("Size: Small")
    else:
        print("Size: Medium")
elif (chestsize>36 and chestsize<40):
    if(shouldersize<55):
        print("Size: Medium")
    else:
        print("Size: Large")
elif (chestsize>40 and chestsize<44):
    if(shouldersize<60):
        print("Size: Large")
    else:
        print("Size: Extra Large")
        
elif (chestsize>44 and chestsize<48):
    if(shouldersize<65):
        print("Size: Extra Large")
    else:
        print("Size: 2-Extra Large")
elif (chestsize>48):
    print("Size: 3-Extra Large")
