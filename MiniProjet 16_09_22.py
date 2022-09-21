#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# récupérer le chemin du répertoire courant
path = os.getcwd()
print("Le répertoire courant est : " + path)
# récupérer le nom du répertoire courant
repn = os.path.basename(path)
print("Le nom du répertoire est : " + repn)


# # ANALYSE EXPLORATOIRE DES DONNES
# ##   Importation des données

# In[2]:


### Importation des données 

import pandas as pd
import numpy as np
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import matplotlib.pyplot as plt 
# %matplotlib inline
import seaborn as sns
from scipy import stats
from scipy.stats.mstats import winsorize


# In[3]:


df= pd.read_csv('file:///C:/Users/DELL/Desktop/Mini_Projet_8/credit_risk_dataset convert.csv' , sep=';')


# # Préparation des données

# In[4]:


# Afficher les 20 premières lignes
print(df.head(15))


# In[5]:


# le nombre de lignes et de colonnes
df.shape


# In[6]:


### Visualisation des données

import seaborn as sns
sns.countplot(x='cb_person_default_on_file',data=df);


# In[7]:


# Nous constatons un grand déséquilibre des classes de la variable cible en défaveur de la modalité Y, un 
# rééchantillonnage sera nécessaire pour l''optimisation de la prévision


# In[8]:


t1=sns.countplot(x='person_home_ownership',data=df);


# In[9]:


sns.countplot(x='loan_intent',data=df);


# In[10]:


t2=sns.countplot(x='loan_grade',data=df);


# In[11]:


# détection des valurs aberrantes


# In[12]:


sns.boxplot(y='person_income',data=df)


# In[13]:


sns.boxplot(y='person_emp_length',data=df)


# In[14]:


sns.boxplot(y='loan_amnt',data=df)


# In[15]:


sns.boxplot(y='loan_int_rate',data=df)


# In[16]:


sns.boxplot(y='person_age',data=df)


# In[17]:


sns.boxplot(y='loan_percent_income',data=df)


# In[18]:


# Presque toutes les variables quantitatives contiennent des valeurs aberrantes, 
# toutefois celles de la variable 'person_income' restent impressionnantes.


# In[19]:



### Rélation entre âge et variable cible

sns.boxplot(x='cb_person_default_on_file',y='person_age',data=df)

# On constate une presque égalité des âges des deux sous groupes ayant réppndus par 'Y'et par 'N'


# In[20]:


# Résumé statistique

df.describe()


# In[21]:


# valeurs manquantes
df.isna().sum()


# In[22]:


# seulement les variables person_emp_length et loan_int_rate ont des valeurs manquantes

df.fillna(value={'person_emp_length':df['person_emp_length'].mean()}, inplace=True)

df.fillna(value={'loan_int_rate':df['loan_int_rate'].mean()}, inplace=True)


# In[23]:


#   Encodage des variables qualitatives


# In[24]:


# Les variables qualitatives sont person_home_ownership, loan_intent, loan_grade et cb_person_default_on_file

df['cb_person_default_on_file'].unique()

df['cb_person_default_on_file'].replace({'Y':1, 'N':0},inplace=True)


# In[25]:


# Traitement des variables qualitatives de plus de deux(2) modalités


# In[26]:


## Cas de la variable person_home_ownership

df['person_home_ownership'].unique()

person_home_ownership_encode=pd.get_dummies(df['person_home_ownership'],drop_first=True) # Suppression de lacolinéarité enre les variables crées

person_home_ownership_encode


# In[27]:


## Cas de la variable loan_intent

df['loan_intent'].unique()

loan_intent_encode=pd.get_dummies(df['loan_intent'],drop_first=True)

loan_intent_encode


# In[28]:


## Cas de la variable loan_grade

df['loan_grade'].unique()

loan_grade_encode=pd.get_dummies(df['loan_grade'],drop_first=True)

loan_grade_encode


# In[29]:


# supression des anciennes variables qualitatives dans le jeu de données

df.drop(['person_home_ownership','loan_intent','loan_grade'],axis=1,inplace=True)

df


# In[30]:


# Intégration des nouvelles variables encodées au jeu de données

df=pd.concat([df,person_home_ownership_encode,loan_intent_encode,loan_grade_encode],axis=1)
df.head(3)


# In[31]:


# Caractéristiques du nouveau jeu de données
df.dtypes


# In[32]:


# Traitement des valeurs aberrantes
# L'écart des valeurs entre le troisième quartile et le maximum confirme une fois de plus l'existence de valeurs 
# extrêmes surtout pour les variables person_age,person_emp_length,loan_amnt et cb_person_cred_hist_length.

features=['person_income','person_emp_length','loan_amnt','loan_int_rate','person_age','loan_percent_income',]
features

sns.set_style('dark')
for col in features:
    plt.figure(figsize=(15,4))
    plt.subplot(131)
    sns.distplot(df[col],label="skew: "+str(np.round(df[col].skew(),2)))
    plt.legend()
    plt.subplot(132)
    sns.boxplot(df[col])
    plt.subplot(133)
    stats.probplot(df[col],plot=plt)
    plt.tight_layout()
    plt.show()


# In[33]:


# On remarque l'existence de valeurs extrêtmes pour les variables 'person_income','person_emp_length','loan_amnt','loan_int_rate','person_age' et 'loan_percent_income'


# In[34]:


# copie de la base dans un nouvel objet df_cap 
df_cap=df.copy()


# In[35]:


# Application de la méthode Z-SCORE pour le traitement des valeurs extrêmes

def zscore_capping(df,cols,thr):

 for col in cols:
        
            mean=df[col].mean()
            std=df[col].std()
            lower_bound=mean-thr*std
            upper_bound=mean+thr*std
        
            df[col]=np.where(df[col]>upper_bound,upper_bound,np.where(df[col]<lower_bound,lower_bound,df[col]))


# In[36]:


np.round(df.describe(),4)


# In[37]:


zscore_capping(df_cap,features,3)


# In[38]:


np.round(df_cap.describe(),4)


# In[39]:


# Vérification de la tail du jeu de donées
df_cap.shape

df.shape


# In[40]:


# Division de la base de données traitée 'df_cap'

# Données d'apprentissage:60%
# Données de test        :20%
# Données de validation  :20%
from sklearn.model_selection import train_test_split
seed=111
X=df_cap.drop('cb_person_default_on_file',axis=1)
y=df_cap['cb_person_default_on_file']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=seed,stratify=y)

X_val,X_test,y_val,y_test=train_test_split(X_test,y_test,test_size=0.5,random_state=seed,stratify=y_test)


# In[41]:


# Vérifions la conservation des proportions de la variable cible dans  les differents jeux de données

y_train.value_counts(normalize=True)

y_test.value_counts(normalize=True)

y_val.value_counts(normalize=True)


# In[42]:


# on observe la même distribution de la variable cible pour toutes les bases ci qui permet d'éviter des erreurs d'interprétations


# In[43]:


# Réglage du problème de déséquilibre des classes


# In[44]:


# Nous constatons que pour la variable cible, la classe 'N' ou '0' est sur-représentée soit 82,36% 
# des observations contre une sous représentation de la classe 'Y' ou '1'


# In[45]:


# Méthode de sur-échantionnage (de la classe 'Y')

from sklearn.utils import resample

X2=X_train
X2['cb_person_default_on_file']=y_train.values
X2.head(3)


# In[46]:


# Définissons les classes minoritaires et majoritaires

minority = X2[X2.cb_person_default_on_file == 1]
majority = X2[X2.cb_person_default_on_file == 0]

minority_upsampled = resample(minority,replace=True,n_samples=len(majority)) # tirage avec remise
minority_upsampled

majority.shape


# In[47]:


# les bases équilibrées majority et minority ont le mêm nombre de lignes et de colonnes


# In[48]:


# Constituons une nouvelle base avec la variable cible équilibrée

basesurech=pd.concat([majority,minority_upsampled])
basesurech

basesurech['cb_person_default_on_file'].value_counts(normalize=True)


# In[49]:


# la variable est dès à présent équilibrée et ne subie aucun problème de sous représentation d'une classe


# In[50]:


# Les caractéristique du jeu d'entraînement de suréchantillonnage deviennent:

X_train_up=basesurech.drop('cb_person_default_on_file',axis=1)
y_train_up=basesurech['cb_person_default_on_file']


# In[51]:


# Méthode de sous-échantionnage (de la classe 'N')
 
majority_downsampled=resample(majority,replace=False,n_samples=len(minority))
majority_downsampled


# In[52]:


# Concaténation

basesousech=pd.concat([majority_downsampled,minority])
basesousech


# In[53]:


# Les caractéristique du jeu d'entraînement de souséchantillonnage deviennent:

X_train_down=basesousech.drop('cb_person_default_on_file',axis=1)
y_train_down=basesousech['cb_person_default_on_file']


# # Modélisation et choix des algorithmes

# In[54]:


# Sélection des variables importantes 
#  (construction d'un modèle de forêt aléatoire)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
rf=RandomForestClassifier(random_state=seed)
rf.fit(X_train_up,y_train_up)
accuracy_score(y_val,rf.predict(X_val))


# In[55]:


# Ce modèle a une propension exacte  à prédire de 83.17%


# In[56]:



    print(X_train_up.columns)
    print(rf.feature_importances_) 


# In[57]:


#Rangement par ordre d'importance
varimp=pd.Series(rf.feature_importances_,index=X_train_up.columns).sort_values(ascending=False)
varimp


# In[58]:


# Illustration par un graphique 
sns.barplot(x=varimp,y=varimp.index)
plt.show()


# In[59]:


# Nous décidons de conserver toutes les varibles (ce qui reste un avantage en machine learning)


# # Evaluation des performances et choix du modèle le plus performant en terme de pouvoir prédictif

# In[60]:


# Le choix des Algorithmes

### Entrainement de modèles et détermination des hyperparamètres

# Régression logistique
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
lr=LogisticRegression(random_state=seed,max_iter=500)
lr_hyp={'C':[0.001,0.01,0.1,1,10,100]}
lr_cv=GridSearchCV(lr,lr_hyp,cv=5)
lr_cv.fit(X_train_up,y_train_up)# entrainement 
print(lr_cv.best_score_)
print(lr_cv.best_estimator_) 


# In[61]:


# Forêt aléatoire ( ou regroupement de modèles d'arbres de décision)
RF=RandomForestClassifier(random_state=seed)
RF_hyp={'n_estimators':[5,10,20,50,100,200],'max_depth':[None,2,5,10,15,20]}
RF_cv=GridSearchCV(RF,RF_hyp,cv=5)
RF_cv.fit(X_train_up,y_train_up) 
print(RF_cv.best_score_)
print(RF_cv.best_estimator_)


# In[62]:


# Bagging Classifier

from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier(random_state=seed)
bc_hyp={'n_estimators':[5,10,20,50,100,200]}
bc_cv=GridSearchCV(bc,bc_hyp,cv=5)
bc_cv.fit(X_train_up,y_train_up) 
print(bc_cv.best_score_)
print(bc_cv.best_estimator_)


# In[63]:


# K Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

## Testons plusieurs valeurs de k


# In[64]:


#n_neighbors=3
model0=KNeighborsClassifier(3)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[65]:


#n_neighbors=4
model0=KNeighborsClassifier(4)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[66]:


#n_neighbors=5
model0=KNeighborsClassifier(5)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[67]:


#n_neighbors=10
model0=KNeighborsClassifier(10)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[68]:


#n_neighbors=15
model0=KNeighborsClassifier(15)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[69]:


#n_neighbors=20
model0=KNeighborsClassifier(20)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[70]:


#n_neighbors=30
model0=KNeighborsClassifier(30)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[71]:


#n_neighbors=40
model0=KNeighborsClassifier(40)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[72]:


#n_neighbors=50
model0=KNeighborsClassifier(50)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[73]:


#n_neighbors=2
model0=KNeighborsClassifier(2)
model0.fit(X_train_up,y_train_up)
print(model0.score(X_train_up,y_train_up))


# In[74]:


# On renomme le modèle knn avec la bonne valeur de k
#n_neighbors=2
model01=KNeighborsClassifier(2)
model01.fit(X_train_up,y_train_up)
model01.best_estimator_=model01.score(X_train_up,y_train_up)
print(model01.best_estimator_)


# In[75]:


# On retiens le meilleur model KNN est celui avec pour valeur k=2


# # L’amélioration des résultats

# In[76]:


# L’accuracy est une métrique de performance qui évalue la capacité d’un modèle de classification à bien prédire à la fois les individus positifs et les individus négatifs. Comme la plupart des métriques, elle est calculée à partir de la matrice de confusion et dépend donc du seuil de classification utilisé pour convertir les probabilités en labels.
# L’accuracy présente de grands avantages en présence de données équilibrées comme c'est le cas ici.


# In[77]:


def model_evaluation(model,features,labels):
    pred=model.predict(features)
    score=accuracy_score(y_val,pred)
    print('Score global du modèle:',round(score,3))


# In[78]:


models=[lr_cv.best_estimator_,RF_cv.best_estimator_,bc_cv.best_estimator_,model01.best_estimator_]
        
for model in models:
    print('Modèle :'+str(model))
    model_evaluation(model,X_val,y_val)
    print('-'*90) 
    


# # La Finalisation et l’enregistrement du modèle final

# In[79]:


# Le meilleur modèle semble le modèle KNN avec k=2 soit le modèle "model01"

accuracy_score(y_test,model01.predict(X_test))

# Toutefois on remarque un grand écart ente les prévisions des données test et celles de validation 


# In[80]:


accuracy_score(y_test,bc_cv.best_estimator_.predict(X_test)) 


# In[81]:


# Le meilleur modèle parait donc le modèle Bagging Classifier


# ### Enregistrement du modèle

# In[82]:


import pickle
pickle.dump(bc, open('bc.pkl', 'wb'))

