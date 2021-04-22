# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:22:54 2021

@author: sylvi

"""

import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.write("""
# Dashboard - Modelo de Machine Learning 

Este app contém um modelo de machine learning para classificação de teor alcoólico de vinhos conforme suas propriedades.
Algoritmo utilizado: Random Forest Classifier.
""")


def main():

    df = load_data()

    page = st.sidebar.selectbox("MENU", ['Página Inicial', 'Exploração', 'Previsão'])

    if page == 'Página Inicial':
        st.title('Classificação de Vinhos')
        st.text('Para mais informações, escolha uma das opções disponíveis no menu ao lado')
        st.dataframe(df)
    elif page == 'Exploração':
        st.title('Análise de variáveis')
        if st.checkbox('Mostrar informações do conjunto de dados'):
            st.dataframe(df.describe())
        
        st.markdown('### Analisando correlação entre variáveis')
        st.text('Correlações:')
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)
        st.text('Efeitos de diferentes variáveis')
        fig = sns.pairplot(df, vars=['magnesium', 'flavanoids', 'nonflavanoid_phenols', 'proline'], hue='alcohol')
        st.pyplot(fig)
    else:
        st.title('Aplicação do modelo')
        model, accuracy = train_model(df)
        st.write('Acurácia: ' + str(accuracy))
        st.markdown('### Previsões')
        st.dataframe(df)
        row_number = st.number_input('Selecionar linha por número', min_value=0, max_value=len(df)-1, value=0)
        st.markdown('#### Resultado previsto - nível de teor alcoólico')
        st.text(model.predict(df.drop(['alcohol'], axis=1).loc[row_number].values.reshape(1, -1))[0])


@st.cache(allow_output_mutation=True)
def train_model(df):
    X = np.array(df.drop(['alcohol'], axis=1))
    y= np.array(df['alcohol'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, model.score(X_test, y_test)

@st.cache
def load_data():
    return pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols','flavanoids', 'nonflavanoid_phenols' ,'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline'], delimiter=",", index_col=False)


if __name__ == '__main__':
    main()


