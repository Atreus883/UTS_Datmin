# Isi file: metode_modul.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def generate_interpretations(cluster_means, overall_means):
    """
    Fungsi untuk menghasilkan teks interpretasi untuk setiap klaster.
    """
    interpretations = {}
    for cluster_id, means in cluster_means.iterrows():
        # Hitung deviasi dari rata-rata keseluruhan
        deviation = means - overall_means
        
        # Temukan 3 karakteristik paling menonjol (paling positif)
        top_chars = deviation.nlargest(3)
        # Temukan 3 karakteristik paling tidak menonjol (paling negatif)
        bottom_chars = deviation.nsmallest(3)

        # Buat deskripsi berdasarkan karakteristik
        description = "Klaster ini memiliki karakteristik yang menonjol sebagai berikut:\n\n"
        description += "**📈 Cenderung Sangat Tinggi pada:**\n"
        for feature, value in top_chars.items():
            description += f"- **{feature.replace('_', ' ').title()}** (rata-rata: {means[feature]:.2f})\n"
        
        description += "\n**📉 Cenderung Sangat Rendah pada:**\n"
        for feature, value in bottom_chars.items():
            description += f"- **{feature.replace('_', ' ').title()}** (rata-rata: {means[feature]:.2f})\n"
            
        # Beri nama/persona klaster berdasarkan nilai rata-rata keseluruhan
        avg_score = means.mean()
        if avg_score == cluster_means.mean(axis=1).max():
            persona = "Persona: Tingkat Stres/Beban Paling Tinggi"
        elif avg_score == cluster_means.mean(axis=1).min():
            persona = "Persona: Tingkat Stres/Beban Paling Rendah"
        else:
            persona = "Persona: Tingkat Stres/Beban Menengah"

        interpretations[cluster_id] = {
            "persona": persona,
            "description": description
        }
        
    return interpretations


def jalankan_analisis(data_file):
    """
    Fungsi ini menerima path file data, melakukan seluruh proses clustering,
    dan mengembalikan hasil analisis serta gambar visualisasi.
    """
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. MUAT DATA
    df = pd.read_csv(data_file)

    # 2. PERSIAPAN DATA
    kolom_numerik = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Angkatan' in kolom_numerik:
        kolom_numerik.remove('Angkatan')

    df_numerik = df[kolom_numerik].copy()

    if df_numerik.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='median')
        df_numerik_imputed = pd.DataFrame(imputer.fit_transform(df_numerik), columns=df_numerik.columns)
    else:
        df_numerik_imputed = df_numerik.copy()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_numerik_imputed)
    
    # Hitung rata-rata keseluruhan SEBELUM clustering untuk perbandingan
    overall_means = df_numerik_imputed.mean()

    # 3. METODE ELBOW
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10).fit(data_scaled)
        wcss.append(kmeans.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
    ax_elbow.plot(K_range, wcss, marker='o', linestyle='-', color='darkblue')
    ax_elbow.set_title('Metode Elbow untuk Menentukan K Optimal', fontsize=14, fontweight='bold')
    ax_elbow.set_xlabel('Jumlah Klaster (K)', fontsize=12)
    ax_elbow.set_ylabel('WCSS', fontsize=12)
    ax_elbow.set_xticks(K_range)
    ax_elbow.grid(True, alpha=0.3)
    
    # 4. K-MEANS CLUSTERING
    K_optimal = 3
    kmeans_final = KMeans(n_clusters=K_optimal, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(data_scaled)
    df['Cluster'] = cluster_labels

    # 5. ANALISIS & INTERPRETASI HASIL KLASTER
    cluster_means = df.groupby('Cluster')[kolom_numerik].mean()
    
    # >>> INI BAGIAN BARU: PANGGIL FUNGSI INTERPRETASI <<<
    interpretasi_klaster = generate_interpretations(cluster_means, overall_means)

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 7))
    sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax_heatmap)
    ax_heatmap.set_title('Heatmap Karakteristik Setiap Klaster', fontsize=14, fontweight='bold')
    ax_heatmap.set_xlabel('Cluster', fontsize=12)
    ax_heatmap.set_ylabel('Pertanyaan Survei', fontsize=12)

    # 6. VISUALISASI PCA
    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(data_scaled)
    
    fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
    scatter = ax_pca.scatter(pca_components[:, 0], pca_components[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
    pca_centroids = pca.transform(kmeans_final.cluster_centers_)
    ax_pca.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='X', s=300, c='red', edgecolors='black', label='Centroid')
    ax_pca.set_title('Visualisasi Klaster Mahasiswa (PCA)', fontsize=14, fontweight='bold')
    ax_pca.set_xlabel('Principal Component 1', fontsize=12)
    ax_pca.set_ylabel('Principal Component 2', fontsize=12)
    fig_pca.colorbar(scatter, ax=ax_pca, label='Cluster')
    ax_pca.legend()
    ax_pca.grid(True, alpha=0.3)

    return {
        "dataframe_awal": df.head(),
        "statistik_deskriptif": df.describe(),
        "kolom_clustering": kolom_numerik,
        "distribusi_klaster": df['Cluster'].value_counts().sort_index(),
        "rata_rata_klaster": cluster_means.round(2),
        "figur_elbow": fig_elbow,
        "figur_heatmap": fig_heatmap,
        "figur_pca": fig_pca,
        "dataframe_hasil": df,
        "interpretasi_klaster": interpretasi_klaster # <-- Mengembalikan hasil interpretasi
    }