# Ganti seluruh isi file app.py Anda dengan kode ini

import streamlit as st
import pandas as pd
from metode_modul import (
    persiapan_data_dan_elbow,
    jalankan_clustering_dan_pca,
    hasilkan_analisis_final
)
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

# Inisialisasi session state
if 'data_siap' not in st.session_state:
    st.session_state.data_siap = False
if 'hasil_perbandingan' not in st.session_state:
    st.session_state.hasil_perbandingan = None

# =======================================================
# SIDEBAR
# =======================================================
with st.sidebar:
    st.title("🎓 Dashboard Analisis Klaster")
    st.info("Aplikasi ini melakukan clustering pada data mahasiswa untuk menemukan pola tingkat stres atau kesejahteraan mental.")
    
    uploaded_file = st.file_uploader("Unggah file data Anda (CSV)", type=["csv"])

    st.markdown("---")
    st.write("Yazid Dahren Fauzan")
    st.write("Nazhifan Zahrawaani Sudrajat")
    st.write("Muhamad Faiz Muzaky")
    st.write("Miftah Rijallul Aziz")

# =======================================================
# HALAMAN UTAMA
# =======================================================
st.title("📊 Analisis Clustering Interaktif pada Data Mahasiswa")

# -------------------------------------------------------
# LANGKAH 1: UPLOAD DAN PERSIAPAN DATA
# -------------------------------------------------------
if uploaded_file is not None and not st.session_state.data_siap:
    with st.spinner('Mempersiapkan data dan menghitung Elbow Method...'):
        # Buat direktori jika belum ada
        if not os.path.exists("data"):
            os.makedirs("data")
        
        data_path = os.path.join("data", uploaded_file.name)
        with open(data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Jalankan fungsi persiapan data
        data_scaled, df_original, kolom_numerik, fig_elbow = persiapan_data_dan_elbow(data_path)
        
        # Simpan hasil ke session state
        st.session_state.data_scaled = data_scaled
        st.session_state.df_original = df_original
        st.session_state.kolom_numerik = kolom_numerik
        st.session_state.fig_elbow = fig_elbow
        st.session_state.data_siap = True
        st.rerun() # Rerun untuk menampilkan bagian selanjutnya
else:
    st.info('Silakan unggah file data CSV melalui menu di sidebar untuk memulai analisis.')

# -------------------------------------------------------
# LANGKAH 2: TAMPILKAN ELBOW & MINTA INPUT PARAMETER
# -------------------------------------------------------
if st.session_state.data_siap:
    st.header("Langkah 1: Menentukan Jumlah Klaster (K-Means)")
    st.write("Gunakan Metode Elbow di bawah untuk mendapatkan intuisi jumlah klaster (K) yang optimal untuk K-Means dan Hierarchical. Titik 'siku' pada grafik adalah kandidat K yang baik.")
    st.pyplot(st.session_state.fig_elbow)

    st.markdown("---")
    st.header("Langkah 2: Atur Parameter & Bandingkan Metode Clustering")
    st.write("Masukkan parameter untuk setiap metode. Setelah itu, klik tombol di bawah untuk melihat perbandingan visualnya menggunakan PCA.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("K-Means")
        k_kmeans = st.number_input("Jumlah Klaster (K)", min_value=2, max_value=10, value=3, key="k_kmeans", help="Pilih jumlah klaster berdasarkan plot Elbow di atas.")

    with col2:
        st.subheader("DBSCAN")
        eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="eps", help="Jarak maksimum antara dua sampel agar dianggap sebagai tetangga.")
        min_samples = st.slider("Min Samples", min_value=1, max_value=20, value=5, key="min_samples", help="Jumlah sampel minimum dalam satu lingkungan agar sebuah titik dianggap sebagai core point.")

    with col3:
        st.subheader("Hierarchical Clustering")
        k_hierarchical = st.number_input("Jumlah Klaster (K)", min_value=2, max_value=10, value=3, key="k_hierarchical", help="Jumlah klaster akhir yang diinginkan.")

    if st.button("Bandingkan Hasil Clustering", type="primary"):
        with st.spinner("Menjalankan semua algoritma clustering..."):
            params_kmeans = {'metode': 'K-Means', 'k': k_kmeans}
            labels_kmeans, fig_pca_kmeans = jalankan_clustering_dan_pca(st.session_state.data_scaled, params_kmeans)

            params_dbscan = {'metode': 'DBSCAN', 'eps': eps, 'min_samples': min_samples}
            labels_dbscan, fig_pca_dbscan = jalankan_clustering_dan_pca(st.session_state.data_scaled, params_dbscan)

            params_hierarchical = {'metode': 'Hierarchical', 'k': k_hierarchical}
            labels_hierarchical, fig_pca_hierarchical = jalankan_clustering_dan_pca(st.session_state.data_scaled, params_hierarchical)
            
            st.session_state.hasil_perbandingan = {
                "K-Means": {"labels": labels_kmeans, "fig": fig_pca_kmeans},
                "DBSCAN": {"labels": labels_dbscan, "fig": fig_pca_dbscan},
                "Hierarchical": {"labels": labels_hierarchical, "fig": fig_pca_hierarchical}
            }

# -------------------------------------------------------
# LANGKAH 3: TAMPILKAN PERBANDINGAN & MINTA PILIHAN FINAL
# -------------------------------------------------------
if st.session_state.get('hasil_perbandingan') is not None:
    st.markdown("---")
    st.header("Visualisasi Perbandingan Hasil Clustering (via PCA)")
    st.write("Perhatikan sebaran klaster dari setiap metode. Pilih metode yang menurut Anda memberikan pemisahan klaster yang paling jelas dan logis.")
    
    vis_col1, vis_col2, vis_col3 = st.columns(3)
    with vis_col1:
        st.pyplot(st.session_state.hasil_perbandingan["K-Means"]["fig"])
    with vis_col2:
        st.pyplot(st.session_state.hasil_perbandingan["DBSCAN"]["fig"])
    with vis_col3:
        st.pyplot(st.session_state.hasil_perbandingan["Hierarchical"]["fig"])
        
    st.markdown("---")
    st.header("Langkah 3: Pilih Metode untuk Analisis Mendalam")
    
    pilihan_final = st.radio(
        "Berdasarkan visualisasi di atas, metode mana yang ingin Anda gunakan untuk analisis interpretasi persona?",
        options=["K-Means", "DBSCAN", "Hierarchical"],
        horizontal=True
    )

    if st.button(f"Lanjutkan Analisis dengan {pilihan_final}"):
        with st.spinner(f"Menghasilkan interpretasi dan analisis akhir untuk {pilihan_final}..."):
            labels_terpilih = st.session_state.hasil_perbandingan[pilihan_final]["labels"]
            
            # Panggil fungsi analisis final
            hasil_akhir = hasilkan_analisis_final(
                st.session_state.df_original.copy(), # Gunakan copy untuk menghindari modifikasi
                labels_terpilih,
                st.session_state.kolom_numerik
            )
            st.session_state.hasil_akhir = hasil_akhir
            st.session_state.metode_terpilih = pilihan_final

# -------------------------------------------------------
# LANGKAH 4: TAMPILKAN HASIL ANALISIS AKHIR
# -------------------------------------------------------
if st.session_state.get('hasil_akhir') is not None:
    st.success(f"Analisis Selesai dengan Metode: **{st.session_state.metode_terpilih}**")
    
    hasil_analisis = st.session_state.hasil_akhir
    
    tab1, tab2, tab3 = st.tabs([
        "🎯 Hasil & Interpretasi", 
        "🎨 Visualisasi Klaster",
        "💾 Data Hasil"
    ])

    with tab1:
        st.header("Analisis dan Interpretasi Hasil Klaster")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Jumlah Entitas per Klaster:**")
            st.dataframe(hasil_analisis["distribusi_klaster"])
            if -1 in hasil_analisis["distribusi_klaster"].index:
                st.caption("Klaster -1 pada DBSCAN menunjukkan data noise (outlier).")

        with col2:
            st.write("**Persentase Distribusi:**")
            total = hasil_analisis['distribusi_klaster'].sum()
            st.dataframe(hasil_analisis["distribusi_klaster"].apply(lambda x: f"{(x/total*100):.1f}%"))

        st.markdown("---")
        st.write("**Rata-rata Nilai Setiap Fitur per Klaster:**")
        st.dataframe(hasil_analisis["rata_rata_klaster"])
        
        st.markdown("---")
        st.header("🕵️‍♂️ Interpretasi Persona Setiap Klaster")
        interpretasi = hasil_analisis["interpretasi_klaster"]
        
        # Urutkan klaster untuk tampilan yang konsisten, kecuali -1
        kunci_klaster = sorted([k for k in interpretasi.keys() if k != -1])
        
        kolom_interpretasi = st.columns(len(kunci_klaster))
        for i, key in enumerate(kunci_klaster):
            with kolom_interpretasi[i]:
                cluster_data = interpretasi[key]
                st.subheader(f"Klaster {key}")
                st.metric(label="Persona Utama", value=cluster_data["persona"].split(": ")[1])
                with st.expander("Lihat Detail Karakteristik"):
                    st.markdown(cluster_data["description"])
    
    with tab2:
        st.header("Visualisasi Hasil Klaster Terpilih")
        st.subheader("Heatmap Karakteristik Klaster")
        st.pyplot(hasil_analisis["figur_heatmap"])
        
    with tab3:
        st.header("Unduh Data Hasil Clustering")
        st.dataframe(hasil_analisis["dataframe_hasil"])
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(hasil_analisis["dataframe_hasil"])
        
        st.download_button(
           label="📥 Unduh Data sebagai CSV",
           data=csv,
           file_name=f'hasil_{st.session_state.metode_terpilih.lower()}_clustering.csv',
           mime='text/csv',
        )