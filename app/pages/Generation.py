import streamlit as st
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.direct_faker import DGenerator
from src.statistical_faker import SGenerator

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sn


df_exists = False
custom_template = {'layout':
                   go.Layout(
                       font = {'family': 'Helvetica',
                               'size': 14,
                               'color': '#1f1f1f'},
                       
                       title = {'font': {'family': 'Helvetica',
                                         'size': 20,
                                          'color': '#1f1f1f'}},
                       
                       legend = {'font': {'family': 'Helvetica',
                                          'size': 14,
                                          'color': '#1f1f1f'}},
                       
                       plot_bgcolor = '#f2f2f2',
                       paper_bgcolor = '#ffffff'
                   )}

st.set_page_config(
    page_title="Data generator",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Data Generator")
cols=st.columns(4)

with cols[0]:
    num_records = st.number_input("# of Records to Generate", min_value=1, max_value=1000000, value=10, step=1)
with cols[1]:
    file_separator = st.selectbox("Field Separator", [",",";",":","|","#"],key=f"separator")
with cols[2]:
    include_header = st.selectbox("Include Header?", ["Yes", "No"])
with cols[3]:
    model_name = st.selectbox("Model Name", ["Direct", "Statistical"])


st.markdown("""----""")

st.sidebar.header("Download Your Dataset")
file_prefix = st.sidebar.text_input("File Prefix", "data.csv")
file_suffix = "" 

if st.sidebar.button("Generate"):
    if num_records <= 0:
        st.sidebar.error("Number of records must be greater than 0.")
    else:
        if model_name == "Direct":
            gen = DGenerator(is_laund=False)
        else:
            gen = SGenerator(is_laund=False)

        df = gen.gen_dataframe(num_records)
        df_exists = True

        file_name = f"{file_prefix}{file_suffix}"
        file_csv = df.to_csv(index=False, header=include_header, sep=file_separator, encoding='utf-8')   
        st.sidebar.success(f"Data generated. Click the button to download {file_name}")
        st.sidebar.download_button(
            label="Download Locally",
            data=file_csv,
            file_name=file_name,
            mime='text/csv',
        )



if df_exists:
    st.header("Generated Data Sample")
    st.table(df.head(10))

    fig_time = px.histogram(df, x = 'Timestamp', title = "<b>Transaction Time Distribution</b>", color_discrete_sequence = ['#FF7F50'])
    fig_fr_bank = px.histogram(df, x = 'From Bank', title = "<b>From which bank transaction is made</b>", color_discrete_sequence = ['#FF7F50'])
    fig_to_bank = px.histogram(df, x = 'To Bank', title = "<b>To which bank transaction is made</b>", color_discrete_sequence = ['#FF7F50'])
    fig_amount = px.histogram(df, x = 'Amount Received', title = "<b>Amount of currency</b>", color_discrete_sequence = ['#FF7F50'])

    fig_curr = px.bar(df.value_counts('Receiving Currency', ascending = False).head(100),
                 x = df.value_counts('Receiving Currency', ascending = False).head(100),
                 y = df.value_counts('Receiving Currency', ascending = False).head(100).index,
                 title = "<b>Currencies</b>",
                 color_discrete_sequence = ['#FF7F50'])

    fig_curr.update_layout(height = 600, width = 1000, template = custom_template, xaxis_title = '<b>Rating count</b>',
                            yaxis_title = '<b>Currencies</b>')

    fig_curr.update_yaxes(automargin = True, title_standoff = 10)

    fig_form = px.bar(df.value_counts('Payment Format', ascending = False).head(10),
                 x = df.value_counts('Payment Format', ascending = False).head(10),
                 y = df.value_counts('Payment Format', ascending = False).head(10).index,
                 title = "<b>Payment Formats</b>",
                 color_discrete_sequence = ['#FF7F50'])

    fig_form.update_layout(height = 600, width = 1000, template = custom_template, xaxis_title = '<b>Rating count</b>',
                            yaxis_title = '<b>Formats</b>')

    fig_form.update_yaxes(automargin = True, title_standoff = 10)

    corr = gen.corr_matrix(df)
    fig_corr, ax = plt.subplots()
    sn.heatmap(corr, ax=ax, annot=True)


    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['**Timestamp**', '**From Bank**', '**To Bank**', '**Amount**', '**Currencies**', '**Payment Format**', '**Correlation Matrix**'])
    with tab1:
        st.header('Timestamps')
        st.plotly_chart(fig_time)

    with tab2:
        st.header('From Bank')
        st.plotly_chart(fig_fr_bank)

    with tab3:
        st.header('To Bank')
        st.plotly_chart(fig_to_bank)

    with tab4:
        st.header('Amount') 
        st.plotly_chart(fig_amount)
        
    with tab5:
        st.header('Used Currencies') 
        st.plotly_chart(fig_curr)

    with tab6:
        st.header('Payment Format') 
        st.plotly_chart(fig_form)

    with tab7:
        st.header('Correlation matrix')
        st.write(fig_corr)
        

else:
    st.warning("Click the generate button to see a preview.")

