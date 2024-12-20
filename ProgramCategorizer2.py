import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os
from io import BytesIO

st.title('Classifier Program Budaya/Kegiatan/Deliverables')
st.write('Klasifikasi Program Budaya/Kegiatan/Deliverables menjadi STRATEGIS, TAKTIKAL, atau OPERASIONAL')

def train_model(df, text_column, label_column):
    # Preprocessing
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[text_column])
    y = df[label_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearSVC()
    model.fit(X_train, y_train)
    
    # Hitung akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, accuracy

def save_model(model, vectorizer):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Sidebar untuk upload dan training
st.sidebar.header('Upload Data Training')
train_file = st.sidebar.file_uploader("Upload file Excel untuk training", type=['xlsx'])

if train_file is not None:
    try:
        train_df = pd.read_excel(train_file)
        st.sidebar.success("File training berhasil diupload!")
        
        # Pilih kolom
        text_column = st.sidebar.selectbox('Pilih kolom teks program/kegiatan:', train_df.columns)
        label_column = st.sidebar.selectbox('Pilih kolom label (STRATEGIS/TAKTIKAL/OPERASIONAL):', train_df.columns)
        
        if st.sidebar.button('Train Model'):
            with st.spinner('Training model...'):
                model, vectorizer, accuracy = train_model(train_df, text_column, label_column)
                save_model(model, vectorizer)
                st.sidebar.success('Model berhasil di-training!')
                st.sidebar.write(f'Akurasi Model: {accuracy:.2%}')
                
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Main area untuk prediksi
st.header('Klasifikasi Data Baru')

# Tab untuk memilih metode input
tab1, tab2 = st.tabs(["Upload File", "Input Manual"])

with tab1:
    pred_file = st.file_uploader("Upload file Excel untuk klasifikasi", type=['xlsx'])
    
    if pred_file is not None:
        try:
            if not (os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl')):
                st.error('Model belum di-training! Silakan upload data training terlebih dahulu.')
            else:
                pred_df = pd.read_excel(pred_file)
                st.success("File berhasil diupload!")
                
                pred_column = st.selectbox('Pilih kolom teks yang akan diklasifikasi:', pred_df.columns)
                
                if st.button('Klasifikasi File'):
                    with st.spinner('Mengklasifikasi data...'):
                        model, vectorizer = load_model()
                        X_pred = vectorizer.transform(pred_df[pred_column])
                        predictions = model.predict(X_pred)
                        
                        pred_df['Hasil_Klasifikasi'] = predictions
                        st.write('Hasil Klasifikasi:')
                        st.dataframe(pred_df)
                        
                        # Perbaikan download button
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            pred_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="Download hasil klasifikasi (Excel)",
                            data=buffer.getvalue(),
                            file_name="hasil_klasifikasi.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    input_text = st.text_area("Tuliskan Program Budaya/Kegiatan/Deliverables Anda di bawah ini:")
    
    if st.button('Klasifikasi Teks'):
        if not input_text:
            st.warning("Mohon masukkan teks terlebih dahulu!")
        elif not (os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl')):
            st.error('Model belum di-training! Silakan upload data training terlebih dahulu.')
        else:
            with st.spinner('Mengklasifikasi teks...'):
                model, vectorizer = load_model()
                X_pred = vectorizer.transform([input_text])
                prediction = model.predict(X_pred)

                
                st.write('Hasil Klasifikasi:')
                st.info(f"Teks termasuk kategori: {prediction}")

# Tambahan informasi
st.markdown("""
### Petunjuk Penggunaan:
Applikasi ini berguna untuk menilai KATEGORI Program Budaya/Aktivitas/Deliverables
1. Pilih metode input:
   - Upload file Excel yang ingin diklasifikasi, atau
   - Masukkan teks secara manual dengan mengetikkan pada kolom yang tersedia
2. Klik 'Klasifikasi' untuk mendapatkan hasil
3. Perhatikan AKURASI MODEL
""")
