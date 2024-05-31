import streamlit as st

st.write("# Тема ВКР: Приложение обработки и анализа синтетических данных")

st.sidebar.success("Выберите режим работы выше")

st.markdown(
    """
        Приложение создано для генерации и поиска табличных синтетических данных, а именно
        для генерации множества транзакции на основе данных генератора [AMLWorld](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
        
        Детектор синтетических данных принимает только файлы в формате .csv со структурой аналогичной наборам данных AMLWorld.
        
        Генератор синтетических данных генерирует данные в том же виде, в котором они представлены в наборах данных AMLWorld.
        
        На данный момент приложение предоставляет две модели генерации и две модели поиска синтетических данных. 
    """
)