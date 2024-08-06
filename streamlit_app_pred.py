import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from xgboost import XGBClassifier
import joblib

MODEL_PATH = "models/best_model_XGBoost.pkl" 
DATA_PATH = "models/df_val_streamlit.parquet"

st.set_page_config(page_title="Opciones de Pago")

@st.cache_resource
def load_data():
    data = pd.read_parquet(DATA_PATH)
    return data

def load_x_y(file_path):
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def calculate_shap(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    return explainer, shap_values_cat_train

def plot_shap_values(model, explainer, shap_values_cat_train, customer_id,  X_train,X_train_vars):
    customer_index = X_train[X_train['llave_modelo'] == customer_id].index[0]
    fig, ax_2 = plt.subplots(figsize=(6,6), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_cat_train[customer_index], X_train[['prob_auto_cura'
                ,'valor_cuota_mes_n-2'
                ,'lote_n-1'
                ,'prob_alrt_temprana']][X_train['llave_modelo'] == customer_id], link="logit")
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_cat_train, X_train):
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12,12))
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()

def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    display_shap_summary(shap_values_cat_train, X_train)

def plot_shap(model, data, customer_id, X_train):
    X_train_vars=X_train[['prob_auto_cura'
                ,'valor_cuota_mes_n-2'
                ,'lote_n-1'
                ,'prob_alrt_temprana']]

    explainer, shap_values_cat_train = calculate_shap(model, X_train_vars)
    

    plot_shap_values(model, explainer, shap_values_cat_train, customer_id, X_train,X_train_vars)


    customer_index = X_train[X_train['llave_modelo'] == customer_id].index[0]
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_train[customer_index],X_train_vars.columns, max_display=20)


st.title("Opciones de Pago")

def main():
    model = load_model()
    data = load_data()

    max_prob_auto_cura = data['prob_auto_cura'].max()
    max_valor_cuota_mes_n_2 = data['valor_cuota_mes_n-2'].max()
    max_lote_n_1 = data['lote_n-1'].max()
    max_prob_alrt_temprana = data['prob_alrt_temprana'].max()


    election = st.radio("Opciones:", 
                        ("Valores de SHAP por usuario",
                          "Calcular la probabilidad de aceptar una opción de pago"))
    available_customer_ids = data['llave_modelo'].tolist()

    if election == "Valores de SHAP por usuario":
        st.write("threshold del modelo: 0.35.")

        customer_id = st.selectbox("Choose the Customer", available_customer_ids)
        customer_index = data[data['llave_modelo'] == customer_id].index[0]

        st.write(f"Cliente {customer_id}: Valor actual de opción de pago : {data['var_rpta_alt'].iloc[customer_index]}")
        st.write(f"Cliente {customer_id}: XGBoost predicción para la opción de pago : {data['pred'].iloc[customer_index]}")
        
        X_train = data[["llave_modelo",'prob_auto_cura'
                ,'valor_cuota_mes_n-2' 
                ,'lote_n-1'
                ,'prob_alrt_temprana']]
        
        plot_shap(model, data, customer_id, X_train=X_train)
    elif election == "Calcular la probabilidad de aceptar una opción de pago":
        customerID = "James Rodriguez 10"
        prob_auto_cura = st.number_input("prob_auto_cura", min_value=0.0, max_value=max_prob_auto_cura , step=0.1)
        valor_cuota_mes_n_2 = st.number_input("valor_cuota_mes_n-2", min_value=0.0, max_value=max_valor_cuota_mes_n_2, step=1000.0)
        lote_n_1 = st.number_input("lote_n-1", min_value=0, max_value=int(max_lote_n_1), step=1)
        prob_alrt_temprana = st.number_input("prob_alrt_temprana", min_value=0.0, max_value=1.0, step=0.1)

        confirmation_button = st.button("Confirmar")
        if confirmation_button:
                new_customer_data = pd.DataFrame({
                    "llave_modelo": [customerID],
                    "prob_auto_cura": [prob_auto_cura],
                    "valor_cuota_mes_n-2": [valor_cuota_mes_n_2],
                    "lote_n-1": [lote_n_1],
                    "prob_alrt_temprana": [prob_alrt_temprana]
                })

                new_customer_data_pred = new_customer_data[['prob_auto_cura'
                ,'valor_cuota_mes_n-2' 
                ,'lote_n-1'
                ,'prob_alrt_temprana']]

                prediction = model.predict_proba(new_customer_data_pred)[:,1]

                formatted_probability = "{:.2%}".format(prediction.item())
                threshold = 0.35
                st.write("threshold del modelo: 0.35.")

                if prediction >= threshold and prediction <= 0.65:
                    big_text = "<h1>La probabilidad de aceptar una opción de pago es moderada.</h1>"
                    st.markdown(big_text, unsafe_allow_html=True)
                elif prediction > 0.65:
                    big_text = "<h1>La probabilidad de aceptar una opción de pago es alta.</h1>"
                    st.markdown(big_text, unsafe_allow_html=True)
                else:
                    big_text = "<h1>La probabilidad de aceptar una opción de pago es baja.</h1>"
                    st.markdown(big_text, unsafe_allow_html=True)

                

                big_text = f"<h1>Probabilidad Opcion de Pago: {formatted_probability}</h1>"
                st.markdown(big_text, unsafe_allow_html=True)
                st.write(new_customer_data.to_dict())


if __name__ == "__main__":
    main()