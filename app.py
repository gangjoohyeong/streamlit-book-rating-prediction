import streamlit as st
# import yaml
# import io
from predict import load_model, get_prediction
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide")

def main():
    st.markdown("<h1 style='text-align: center; color: black;'>Book Rating Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: grey;'>Using Streamlit by Joohyeong</h4>", unsafe_allow_html=True)
    
    ### file upload  (test_ratings.csv)
    uploaded_file = st.file_uploader("Choose a CSV", type=['csv'])
    
    
    if uploaded_file:
        ### uploaded_file -> read_csv
        test = pd.read_csv(uploaded_file, encoding='utf-8')
        result = test.copy()
        
        ### model load & evalutation
        data, model = load_model(test)
        model.eval()
        
        ### preview
        example = uploaded_file.getvalue().decode('utf-8').splitlines()
        st.session_state['preview'] = ''
        for i in range(0, min(5, len(example))):
            st.session_state['preview'] += (example[i] + '\n')
        st.text_area("CSV Preview (5 Lines)", "", height=150, key="preview")
        
        # button click -> prediction start
        if st.button('Start prediction'):
            st.write('Predicting Book Rating ...')
            predicts = get_prediction(model, data)
            
            result['rating'] = predicts
            st.write(result)
            
            
            # button for download results
            @st.cache_data
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')
            
            st.download_button(
            "Download Results",
            convert_df(result),
            f"prediction_ratings_{datetime.now()}.csv",
            "text/csv",
            key='download-csv'
            )
        
main()