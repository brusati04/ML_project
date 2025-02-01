import pandas as pd
import numpy as np
import itertools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate, RepeatedStratifiedKFold

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_openml
from scipy.stats import loguniform, beta, uniform

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as IMBPipeline

import missingno as msno

import matplotlib.pyplot as plt
import warnings

# warnings.filterwarnings('ignore')



# Carichiamo il dataset
dataset = pd.read_csv('dataset_1.csv')


# if input("do you whant to finish quickly the project?(y/n):") == "y":
#     dataset.dropna(axis=1)
#     dataset.info()
#     print("well done, now 30L")
# else:
#     pass


# Analizziamo il bilanciamento della variabile target
sm = dataset["smoking"].value_counts(normalize=True)
print(f"Distribuzione della variabile target: {sm}")

# analizziamo la distribuzione dei Nan values:
for col in dataset:
    Nan=dataset[col].isnull().sum()
    print(f"{col} has: {Nan} nan values")

    # Nan = dataset[col].size - dataset[col].value_counts().sum()
    # print(f"missing values in {col}: {Nan}")

# msno.matrix(dataset)
# plt.show()

n = 2  # Set your threshold
num_rows = (dataset.isna().sum(axis=1) > n).sum()
print(f"Number of rows with more than {n} NaN values: {num_rows}")


# dataset.dropna(axis=0, thresh=len(dataset.columns)-5)


# The columns not reported in the figure will be discarded.

# For features age and fare the pipeline is composed by two transformers:

# KNNImputer: both features contain missing values, so we have to apply an imputation strategy. In this case the strategy is based on the idea of 
# k
# k-nearest neighbors.
# StandardScaler: both features are numerical
# For features pclass:

# An OrdinalEncoder transforms the strings '3','2' and '1' corresponding to the ticket classes into the numerical values 3,2 and 1.
# For features sex and embarked, we apply:

# SimpleImputer: feature embarked contains two missing values, while the column sex will be untouched. As a strategy we use 'most_frequent' since both features are categorical
# OneHotEncoder: features are categorical.
# For features sbsp and parch we define a customer transformer that builds a new feature is_alone indicating whether the passenger travelled alone or not. More details about how to code customer transformers in the following optional section.

# For feature name we define a further customer transformer to infer the title (Mr, Miss, Doc, Captain, etc..) from the fullname.

# ID: drop
# gender: ordinal, 
# age: minmax,
# oral: ordinal
# dental caries: none
# tartar: ordinal
# O/W: standardization x 20, 


minmax_age = MinMaxScaler()
oe_oral = Pipeline([
        ("pipe_sim", SimpleImputer(strategy="most_frequent")),
        ("pipe_ord",  OrdinalEncoder(categories=[["N","Y"]]))
        ])
oe_tartar = Pipeline([
        ("pipe_sim", SimpleImputer(strategy="most_frequent")),
        ("pipe_ord",  OrdinalEncoder(categories=[["N","Y"]]))
        ])
oe_gender = Pipeline([
        ("pipe_sim", SimpleImputer(strategy="most_frequent")),
        ("pipe_ord", OrdinalEncoder(categories=[["F","M"]]))
        ])
std_body_signals = Pipeline([
        ("pipe_sim", SimpleImputer(strategy="most_frequent")),
        ("pipe_std", StandardScaler())
        ])
# Separiamo feature e target columns:
X = dataset.drop(columns=["smoking"])
y = dataset["smoking"]

# Divisione in train e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  stratify = y, random_state=42, shuffle=True)


# COLUMN TRASFORMATION
body_signals = [
    "height(cm)","weight(kg)","waist(cm)","eyesight(left)","eyesight(right)",
    "hearing(left)","hearing(right)","systolic","relaxation","fasting blood sugar",
    "Cholesterol","triglyceride","HDL","LDL","hemoglobin","Urine protein",
    "serum creatinine","AST","ALT","Gtp"
]

smoking_tr = ColumnTransformer(
    transformers=[
        ("id", "drop", ["ID"]),
        ("gender", oe_gender, ["gender"]),
        ("age", minmax_age, ["age"]),
        ("body_signals", std_body_signals, body_signals),
        ("oral", oe_oral, ["oral"]),
        ("tartar", oe_tartar, ["tartar"])
    ],
    verbose_feature_names_out= False,
    remainder = "passthrough",
    sparse_threshold = 1
)

X_train = smoking_tr.fit_transform(X_train)
X_test = smoking_tr.fit_transform(X_test)

modello = input("scrivi il modello da utilizare: ")
match modello:

    case "logistic_regression":
        # Creiamo un modello Logistic Regression e lo addestriamo
        model_lr = LogisticRegression(max_iter=1000, random_state=42)
        model_lr.fit(X_train, y_train)

        # Effettuiamo predizioni sul test set
        y_pred_lr = model_lr.predict(X_test)

        # Valutiamo le performance del modello
        print("Report Logistic Regression:")
        print(classification_report(y_test, y_pred_lr))

        print("Matrice di Confusione Logistic Regression:")
        print(confusion_matrix(y_test, y_pred_lr))
        print("Accouracy: ", accuracy_score(y_test, y_pred_lr))

    case "random_forest":
        # Creiamo il modello Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf_model.fit(X_train, y_train)

        # Effettuiamo predizioni sul test set con Random Forest
        y_pred_rf = rf_model.predict(X_test)

        # Valutiamo le performance del modello Random Forest
        print("Report Random Forest:")
        print(classification_report(y_test, y_pred_rf))

        print("Matrice di Confusione Random Forest:")
        print(confusion_matrix(y_test, y_pred_rf))
        
        print("Accouracy: ", accuracy_score(y_test, y_pred_rf))
   
    case "decision_tree":
        # Creiamo il modello Random Forest
        dtc_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        dtc_model.fit(X_train, y_train)

        # Effettuiamo predizioni sul test set con Random Forest
        y_pred_dtc = dtc_model.predict(X_test)

        # Valutiamo le performance del modello Random Forest
        print("Report Decision Tree:")
        print(classification_report(y_test, y_pred_dtc))

        print("Matrice di Confusione Decision Tree:")
        print(confusion_matrix(y_test, y_pred_dtc))

        print("Accouracy: ", accuracy_score(y_test, y_pred_dtc))


    case "perceptron":
            # Creiamo un modello Perceptron e lo addestriamo
            model_perceptron = Perceptron(max_iter=1000, random_state=42)
            model_perceptron.fit(X_train, y_train)

            # Effettuiamo predizioni sul test set
            y_pred_perceptron = model_perceptron.predict(X_test)

            # Valutiamo le performance del modello
            print("Report Perceptron:")
            print(classification_report(y_test, y_pred_perceptron))

            print("Matrice di Confusione Perceptron:")
            print(confusion_matrix(y_test, y_pred_perceptron))
            print("Accouracy: ", accuracy_score(y_test, y_pred_perceptron))

