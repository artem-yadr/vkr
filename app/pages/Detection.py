import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.detector import Detector

df_exists = False


st.set_page_config(
    page_title="Synthetic Data Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Synthetic Data Detector")
cols=st.columns(4)

with cols[0]:
    model_name = st.selectbox("Model Name", ["Random Forest Regression", "MLPClassifier"])


st.markdown("""----""")

uploaded_file = st.file_uploader("Choose a file")
df = []
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

st.sidebar.header("Download Results")
file_prefix = st.sidebar.text_input("File Prefix", "data.csv")
file_suffix = "" 
file_name = f"{file_prefix}{file_suffix}"

if st.sidebar.button("Detect"):
    if model_name == "Random Forest Regression":
        detector = Detector(mod='rfr')
    elif model_name == "MLPClassifier":
        detector = Detector(mod='mlp')

    pred = detector.check_df(df)

    my_array = np.array(pred)

    res_df = pd.DataFrame(my_array, columns=['Is Synthetic'])
    df_exists = True

    file_csv = res_df.to_csv(index=False, encoding='utf-8')
    st.sidebar.success(f"Data scanned. Click the button to download the results {file_name}")
    st.sidebar.download_button(
        label="Download Locally",
        data=file_csv,
        file_name=file_name,
        mime='text/csv',
    )

if df_exists:
    tab1, tab2= st.tabs(['**Dataset**', '**Results for given dataset**'])

    with tab1:
        st.header('Preview of given dataset')
        st.table(df.head(10))

    with tab2:
        st.header('Preview of anomaly detection results')
        st.table(res_df.head(10))
else:
    st.warning("Click the detect button to see a preview.")

