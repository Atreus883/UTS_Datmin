import streamlit as st
import pandas as pd
from metode_modul import jalankan_analisis
import os

# =======================================================
# KONFIGURASI HALAMAN
# =======================================================
st.set_page_config(
    page_title="Aplikasi Clustering Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================================================
# SIDEBAR
# =======================================================
with st.sidebar:
    st.title("🎓 Dashboard Analisis Klaster")
    st.info("Aplikasi ini melakukan clustering pada data mahasiswa untuk menemukan pola tingkat stres atau kesejahteraan mental.")
    
    # Upload file
    uploaded_file = st.file_uploader("Unggah file data Anda (CSV)", type=["csv"])

    st.markdown("---")
    st.write("Yazid Dahren Fauzan")
    st.write("Nazhifan Zahrawaani Sudrajat")
    st.write("Muhamad Faiz Muzaky")
    st.write("Miftah Rijallul Aziz")


# =======================================================
# HALAMAN UTAMA
# =======================================================
st.title("📊 Analisis Clustering K-Means pada Data Mahasiswa")
st.write("Silakan unggah file data CSV melalui menu di sidebar untuk memulai analisis.")

# Cek apakah file sudah diunggah
if uploaded_file is not None:
    
    # Tampilkan loading spinner saat analisis berjalan
    with st.spinner('Sedang memproses dan menganalisis data... Mohon tunggu...'):
        # Simpan file yang diunggah sementara
        data_path = os.path.join("data", uploaded_file.name)
        with open(data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Jalankan fungsi analisis dari modul
        hasil_analisis = jalankan_analisis(data_path)
    
    st.success("Analisis Selesai!")

    # Tampilkan hasil analisis menggunakan tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Eksplorasi Data", 
        "📈 Metode Elbow", 
        "🎯 Hasil Clustering", 
        "🎨 Visualisasi Klaster",
        "💾 Data Hasil"
    ])

    with tab1:
        st.header("1. Eksplorasi Data Awal")
        st.write("**Lima Baris Pertama Data:**")
        st.dataframe(hasil_analisis["dataframe_awal"])
        
        st.write("**Statistik Deskriptif:**")
        st.dataframe(hasil_analisis["statistik_deskriptif"])

        st.write("**Kolom yang Digunakan untuk Clustering:**")
        st.write(hasil_analisis["kolom_clustering"])

    with tab2:
        st.header("2. Menentukan Jumlah Klaster Optimal (Metode Elbow)")
        st.write("Grafik di bawah ini menunjukkan WCSS (Within-Cluster Sum of Squares) untuk berbagai jumlah klaster (K). Titik 'siku' pada grafik (umumnya di K=3 atau K=4) adalah jumlah klaster yang optimal.")
        st.pyplot(hasil_analisis["figur_elbow"])

# Ganti bagian 'with tab3:' di file app.py Anda

    with tab3:
        st.header("3. Analisis dan Interpretasi Hasil Klaster")
        st.write("Dengan K optimal yang dipilih (K=3), berikut adalah distribusi mahasiswa per klaster.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Jumlah Mahasiswa per Klaster:**")
            st.dataframe(hasil_analisis["distribusi_klaster"])
        with col2:
            st.write("**Persentase Distribusi:**")
            st.dataframe(hasil_analisis["distribusi_klaster"].apply(lambda x: f"{(x/hasil_analisis['distribusi_klaster'].sum()*100):.1f}%"))

        st.markdown("---")
        st.write("**Rata-rata Nilai Setiap Fitur per Klaster:**")
        st.write("Tabel ini menunjukkan karakteristik utama dari setiap klaster. Nilai yang lebih tinggi menunjukkan tingkat persetujuan/frekuensi yang lebih tinggi terhadap pertanyaan survei.")
        st.dataframe(hasil_analisis["rata_rata_klaster"])
        
        st.markdown("---")
        st.header("🕵️‍♂️ Interpretasi Persona Setiap Klaster")
        st.write("Analisis ini membandingkan rata-rata setiap klaster dengan rata-rata keseluruhan untuk menemukan karakteristik yang paling unik dan menonjol.")

        interpretasi = hasil_analisis["interpretasi_klaster"]
        
        # Buat kolom sebanyak jumlah klaster
        kolom_interpretasi = st.columns(len(interpretasi))

        for i, col in enumerate(kolom_interpretasi):
            with col:
                # Ambil data untuk klaster saat ini
                cluster_data = interpretasi[i]
                
                # Gunakan st.metric untuk menyorot persona
                st.subheader(f"Klaster {i}")
                st.metric(label="Persona Utama", value=cluster_data["persona"].split(": ")[1])
                
                # Tampilkan deskripsi detail
                with st.expander("Lihat Detail Karakteristik"):
                    st.markdown(cluster_data["description"])
        
    with tab4:
        st.header("4. Visualisasi Hasil Klaster")
        
        st.subheader("Heatmap Karakteristik Klaster")
        st.write("Heatmap ini memvisualisasikan tabel rata-rata di atas. Warna yang lebih terang menunjukkan nilai rata-rata yang lebih tinggi, memudahkan untuk membandingkan karakteristik antar klaster.")
        st.pyplot(hasil_analisis["figur_heatmap"])
        
        st.markdown("---")
        
        st.subheader("Visualisasi Klaster dengan PCA (2 Komponen Utama)")
        st.write("Data direduksi menjadi 2 dimensi menggunakan PCA untuk memvisualisasikan sebaran klaster. Titik yang berdekatan memiliki karakteristik yang mirip.")
        st.pyplot(hasil_analisis["figur_pca"])

    with tab5:
        st.header("5. Unduh Data Hasil Clustering")
        st.write("Dataframe di bawah ini adalah data asli yang telah ditambahkan kolom 'Cluster' sebagai hasil analisis.")
        st.dataframe(hasil_analisis["dataframe_hasil"])
        
        # Fungsi untuk mengonversi DataFrame ke CSV
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(hasil_analisis["dataframe_hasil"])
        
        st.download_button(
           label="📥 Unduh Data sebagai CSV",
           data=csv,
           file_name='hasil_clustering_mahasiswa.csv',
           mime='text/csv',
        )
        
else:
    st.info('Menunggu file CSV diunggah. Pastikan format file sesuai.')