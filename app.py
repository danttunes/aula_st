import streamlit as st
import pickle
import numpy as np

#Exercício: Montar uma interface no Streamlit para classificar uma pessoa como portadora de diabetes ou não.
#O modelo de machine learning já está treinado e salvo em um arquivo chamado "trained_model.sav".

#os dados de entrada são:
#1. Número de vezes grávida
#2. Concentração de glicose
#3. Pressão sanguínea
#4. Espessura da pele
#5. Insulina
#6. IMC
#7. Função de pedigree de diabetes
#8. Idade

#todos esses dados são numéricos

#o input do modelo deve ser um array numpy 2d com todas features listadas acima nessa ordem

#o modelo deve retornar 0 ou 1
#se o resultado for 1, a pessoa é portadora de diabetes
#se o resultado for 0, a pessoa não é portadora de diabetes


def load_model():
    model = pickle.load(open('trained_model.sav', 'rb'))
    return model

def main():
    model = load_model()

    st.title("Previsão de Portador de Diabetes")

    st.write("Este aplicativo utiliza os dados que devem ser inseridos abaixo para classificar uma pessoa como portadora de diabetes ou não.")

    st.subheader("Por favor, preencha as informações abaixo:")

    gravidez = st.number_input('Números de vezes grávida:', min_value=0, max_value=10, step=1)
    glicose = st.number_input('Concentração de glicose:', min_value=0, step=1)
    pressao = st.number_input('Pressão sanguínea', min_value=0, step=1)
    pele = st.number_input('Espessura da pele', min_value=0, step=1)
    insulina = st.number_input('Insulina', min_value=0, step=1)
    imc = st.number_input('IMC', min_value=0.0)
    diabetes = st.number_input('Função de pedigree de diabetes',  min_value=0.000, format="%0.3f", step=0.001)
    idade = st.number_input('Idade', min_value=0, step=1)

    button = st.button("Prever Diabetes")

    if button:
        input = np.array([[gravidez, glicose, pressao, pele, insulina, imc, diabetes, idade,]])
        input = input.astype(float)

        prediction = model.predict(input)

        if prediction == 1:
            st.subheader("Essa pessoa é portadora de diabetes")
        if prediction == 0:
            st.subheader("Essa pessoa não é portadora de diabetes")



if __name__ == '__main__':
    main()