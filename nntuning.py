import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, cross_validate

st.title("Neural Network")
st.header('Multi-layer Perceptron hyperparameter tuning')

st.subheader('Titanic Dataset')
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
df[df.columns] = df[df.columns].fillna(df[df.columns].median())
st.write(df)

st.subheader('Variables')
col1, col2, col3 = st.columns(3)

with col1:
    age = st.checkbox('Age', True)
    _class = st.checkbox('class', True)
    sex = st.checkbox('Sex', True)

with col2:
    siblings = st.checkbox('Siblings', True)
    parents = st.checkbox('Parents', True)
    adult_male = st.checkbox('Adult male', True)

with col3:
    fare = st.checkbox('Fare (£)', True)
    embarked = st.checkbox('Embarked port', True)
    who = st.checkbox('Who (male/child)', True)

columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'who', 'adult_male']
cb_columns = [_class, sex, age, siblings, parents, fare, embarked, who, adult_male]
pred_cols = [columns[i] for i, cb in enumerate(cb_columns) if cb]

kfold_ = st.select_slider('KFold', options=np.arange(3, 11), value=5)

hl_amount = st.select_slider('Number of hidden layers', options=np.arange(1, 10), value=2)
hl_size = st.select_slider('Hidden layers size', options=np.arange(1, 50), value=5)
activation = st.selectbox('Activation function', ['relu', 'identity', 'logistic', 'tanh'])
alpha = st.select_slider('Learning rate', options=np.arange(0.0001, 0.1, 0.0001), value=0.001)
learning_rate = st.selectbox('Learning rate', options=['constant', 'invscaling', 'adaptive'])
max_iter = st.select_slider('Maximum number of iterations', options=np.arange(10, 1000), value=50)

mlp_params = {'hidden_layer_sizes': (hl_amount, hl_size), 'activation': activation, 'alpha': float(alpha),
              'learning_rate': learning_rate, 'max_iter': int(max_iter)}

X = None

for col in pred_cols:

    coldata = df[col].to_numpy()

    if col in ('sex', 'embarked', 'who', 'adult_male'):
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(coldata)
        if len(set(encoded)) > 1:
            hot_encoder = OneHotEncoder()
            coldata = hot_encoder.fit_transform(encoded.reshape(-1, 1)).toarray()
        else:
            coldata = encoded

    X = coldata if X is None else np.column_stack((X, coldata))

y = df['survived'].to_numpy()

kf = KFold(n_splits=kfold_, shuffle=True)
scoring_methods = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
cv = cross_validate(MLPClassifier(**mlp_params), X, y, cv=kf, scoring=scoring_methods, return_estimator=True)

scores = {met: np.mean(cv[f'test_{met}']) for met in scoring_methods}
loss_curves = [estimator.loss_curve_ for estimator in cv['estimator']]
max_len_loss_curve = max(len(loss_curve) for loss_curve in loss_curves)
for loss_curve in loss_curves:
    loss_curve.extend([loss_curve[-1]] * (max_len_loss_curve - len(loss_curve)))
loss_dict = dict(enumerate(loss_curves))

df_loss_curves = pd.DataFrame(loss_dict)

st.subheader('Model performances:')
acc_col, f1_col, prec_col, rec_col, bal_col = st.columns(5)
cols = [acc_col, f1_col, prec_col, rec_col, bal_col]
for key, col in zip(scores.keys(), cols):
    col.metric(key, f'{np.round(scores[key]*100, 2)} %')

st.subheader('Loss curves:')
st.line_chart(df_loss_curves)
