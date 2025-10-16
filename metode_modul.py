# Ganti seluruh isi file metode_modul.py Anda dengan kode ini

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# ==============================================================================
# FUNGSI PREPROCESSING (TIDAK BERUBAH)
# ==============================================================================

def manual_imputer(df, strategy='median'):
    """
    Fungsi untuk menangani nilai yang hilang (missing values) secara manual.
    """
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() > 0:
            if strategy == 'median':
                fill_value = df_imputed[col].median()
            elif strategy == 'mean':
                fill_value = df_imputed[col].mean()
            else: # Default to median
                fill_value = df_imputed[col].median()
            df_imputed[col].fillna(fill_value, inplace=True)
    return df_imputed

def manual_scaler(data):
    """
    Fungsi untuk melakukan standardisasi data secara manual.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1 # Hindari pembagian dengan nol
    return (data - mean) / std

# ==============================================================================
# IMPLEMENTASI ALGORITMA CLUSTERING MANUAL
# ==============================================================================

def _euclidean_distance(p1, p2):
    """Helper fungsi untuk menghitung jarak Euclidean."""
    return np.sqrt(np.sum((p1 - p2)**2))

class ManualKMeans:
    """
    Kelas untuk implementasi algoritma K-Means secara manual.
    """
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            for point_idx, point in enumerate(X):
                distances = [_euclidean_distance(point, centroid) for centroid in self.cluster_centers_]
                closest_centroid_idx = np.argmin(distances)
                clusters[closest_centroid_idx].append(point)

            old_centroids = self.cluster_centers_.copy()

            for cluster_idx, cluster_points in enumerate(clusters):
                if cluster_points: # Hindari error jika ada klaster kosong
                    self.cluster_centers_[cluster_idx] = np.mean(cluster_points, axis=0)

            if np.all(self.cluster_centers_ == old_centroids):
                break
        
        self._calculate_inertia(X)
        return self

    def predict(self, X):
        labels = []
        for point in X:
            distances = [_euclidean_distance(point, centroid) for centroid in self.cluster_centers_]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def _calculate_inertia(self, X):
        total_wcss = 0
        labels = self.predict(X)
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroid = self.cluster_centers_[i]
                total_wcss += np.sum((cluster_points - centroid)**2)
        self.inertia_ = total_wcss


class ManualDBSCAN:
    """
    Kelas untuk implementasi algoritma DBSCAN secara manual.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _get_neighbors(self, X, point_index):
        neighbors = []
        for i, point in enumerate(X):
            if _euclidean_distance(X[point_index], point) < self.eps:
                neighbors.append(i)
        return neighbors

    def fit_predict(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1) # -1: Noise
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1: # Sudah diproses
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                # Tandai sebagai noise, bisa jadi border point nanti
                continue
            
            # Titik i adalah core point, mulai klaster baru
            self.labels_[i] = cluster_id
            
            # Gunakan deque untuk efisiensi
            seed_set = deque(neighbors)
            
            while seed_set:
                current_point_idx = seed_set.popleft()

                if self.labels_[current_point_idx] == -1: # Sebelumnya noise atau belum dikunjungi
                    self.labels_[current_point_idx] = cluster_id
                
                # Jika belum dikunjungi, proses
                elif self.labels_[current_point_idx] != -1:
                    continue

                self.labels_[current_point_idx] = cluster_id
                
                # Cek apakah ini juga core point
                current_neighbors = self._get_neighbors(X, current_point_idx)
                if len(current_neighbors) >= self.min_samples:
                    # Tambahkan tetangganya ke seed set
                    for neighbor_idx in current_neighbors:
                         if self.labels_[neighbor_idx] == -1:
                            seed_set.append(neighbor_idx)
            
            cluster_id += 1
            
        return self.labels_


class ManualHierarchicalClustering:
    """
    Kelas untuk implementasi Hierarchical Agglomerative Clustering manual.
    Menggunakan single linkage.
    """
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.labels_ = None

    def _calculate_distance_matrix(self, X):
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = _euclidean_distance(X[i], X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # Setiap titik data adalah satu klaster pada awalnya
        clusters = [[i] for i in range(n_samples)]
        
        dist_matrix = self._calculate_distance_matrix(X)
        dist_matrix[dist_matrix == 0] = np.inf # Ganti 0 dengan infinity

        while len(clusters) > self.n_clusters:
            # Cari pasangan klaster terdekat
            min_dist = np.inf
            merge_idx = (0, 0)
            
            # Cari indeks dari nilai minimum
            row_idx, col_idx = np.unravel_index(np.argmin(dist_matrix, axis=None), dist_matrix.shape)
            
            # Gabungkan dua klaster
            cluster1_indices = clusters[row_idx]
            cluster2_indices = clusters[col_idx]
            
            # Buat klaster baru
            new_cluster = cluster1_indices + cluster2_indices
            
            # Hapus klaster lama, pastikan menghapus indeks yang lebih besar dulu
            if row_idx > col_idx:
                clusters.pop(row_idx)
                clusters.pop(col_idx)
            else:
                clusters.pop(col_idx)
                clusters.pop(row_idx)
                
            clusters.append(new_cluster)

            # Update distance matrix (single linkage)
            new_dist_row = np.full(dist_matrix.shape[0], np.inf)
            # Ambil nilai minimum dari baris/kolom yang digabung
            for i in range(dist_matrix.shape[0]):
                new_dist_row[i] = min(dist_matrix[row_idx, i], dist_matrix[col_idx, i])

            # Hapus baris dan kolom lama
            dist_matrix = np.delete(dist_matrix, [row_idx, col_idx], axis=0)
            dist_matrix = np.delete(dist_matrix, [row_idx, col_idx], axis=1)

            # Tambahkan baris dan kolom baru untuk klaster baru
            new_size = dist_matrix.shape[0]
            new_dist_matrix = np.full((new_size + 1, new_size + 1), np.inf)
            new_dist_matrix[:new_size, :new_size] = dist_matrix
            
            # Hapus nilai yang berhubungan dengan klaster yang digabung
            new_dist_row = np.delete(new_dist_row, [row_idx, col_idx])
            
            new_dist_matrix[new_size, :new_size] = new_dist_row
            new_dist_matrix[:new_size, new_size] = new_dist_row
            
            dist_matrix = new_dist_matrix
        
        # Buat label akhir
        self.labels_ = np.zeros(n_samples, dtype=int)
        for i, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels_[point_idx] = i
        
        return self.labels_

# ==============================================================================
# IMPLEMENTASI PCA MANUAL (TIDAK BERUBAH)
# ==============================================================================

class ManualPCA:
    """
    Kelas untuk implementasi Principal Component Analysis (PCA) secara manual.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components_ = eigenvectors[0:self.n_components]
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# ==============================================================================
# FUNGSI-FUNGSI UTAMA UNTUK ANALISIS (DIRESTRUKTURISASI)
# ==============================================================================

def persiapan_data_dan_elbow(data_file):
    """
    Fungsi ini melakukan preprocessing data dan menghasilkan plot Elbow.
    Mengembalikan data yang sudah di-scaling, nama kolom, dan figure elbow.
    """
    df = pd.read_csv(data_file)
    kolom_numerik = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Angkatan' in kolom_numerik:
        kolom_numerik.remove('Angkatan')

    df_numerik = df[kolom_numerik].copy()

    if df_numerik.isnull().sum().sum() > 0:
        df_numerik_imputed = manual_imputer(df_numerik, strategy='median')
    else:
        df_numerik_imputed = df_numerik.copy()

    data_scaled = manual_scaler(df_numerik_imputed.values)
    
    # Hitung WCSS untuk Elbow Method
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = ManualKMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
    ax_elbow.plot(K_range, wcss, marker='o', linestyle='-', color='darkblue')
    ax_elbow.set_title('Metode Elbow untuk Menentukan K Optimal', fontsize=14, fontweight='bold')
    ax_elbow.set_xlabel('Jumlah Klaster (K)', fontsize=12)
    ax_elbow.set_ylabel('WCSS', fontsize=12)
    ax_elbow.set_xticks(K_range)
    ax_elbow.grid(True, alpha=0.3)
    
    return data_scaled, df, kolom_numerik, fig_elbow

def jalankan_clustering_dan_pca(data_scaled, params):
    """
    Menjalankan algoritma clustering yang dipilih dan membuat visualisasi PCA.
    """
    metode = params['metode']
    
    if metode == 'K-Means':
        model = ManualKMeans(n_clusters=params['k'], random_state=42)
        labels = model.fit_predict(data_scaled)
    elif metode == 'DBSCAN':
        model = ManualDBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = model.fit_predict(data_scaled)
    elif metode == 'Hierarchical':
        model = ManualHierarchicalClustering(n_clusters=params['k'])
        labels = model.fit_predict(data_scaled)
    else:
        raise ValueError("Metode tidak dikenal")

    # Visualisasi PCA
    pca = ManualPCA(n_components=2)
    pca_components = pca.fit_transform(data_scaled)
    
    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
    scatter = ax_pca.scatter(pca_components[:, 0], pca_components[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    
    # Khusus K-Means, tampilkan centroid
    if metode == 'K-Means' and hasattr(model, 'cluster_centers_'):
         pca_centroids = pca.transform(model.cluster_centers_)
         ax_pca.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='X', s=300, c='red', edgecolors='black', label='Centroid')
         ax_pca.legend()

    ax_pca.set_title(f'Visualisasi Klaster ({metode})', fontsize=14, fontweight='bold')
    ax_pca.set_xlabel('Principal Component 1', fontsize=12)
    ax_pca.set_ylabel('Principal Component 2', fontsize=12)
    fig_pca.colorbar(scatter, ax=ax_pca, label='Cluster')
    ax_pca.grid(True, alpha=0.3)

    return labels, fig_pca


def generate_interpretations(cluster_means, overall_means):
    """
    Fungsi untuk menghasilkan teks interpretasi untuk setiap klaster.
    (Tidak diubah)
    """
    interpretations = {}
    for cluster_id, means in cluster_means.iterrows():
        # Abaikan klaster noise (-1) dari DBSCAN
        if cluster_id == -1:
            continue
            
        deviation = means - overall_means
        top_chars = deviation.nlargest(3)
        bottom_chars = deviation.nsmallest(3)

        description = "Klaster ini memiliki karakteristik yang menonjol sebagai berikut:\n\n"
        description += "**📈 Cenderung Sangat Tinggi pada:**\n"
        for feature, value in top_chars.items():
            description += f"- **{feature.replace('_', ' ').title()}** (rata-rata: {means[feature]:.2f})\n"
        
        description += "\n**📉 Cenderung Sangat Rendah pada:**\n"
        for feature, value in bottom_chars.items():
            description += f"- **{feature.replace('_', ' ').title()}** (rata-rata: {means[feature]:.2f})\n"
            
        avg_score = means.mean()
        
        # Cari klaster dengan rata-rata skor tertinggi dan terendah
        non_noise_means = cluster_means.drop(-1, errors='ignore').mean(axis=1)
        
        if avg_score == non_noise_means.max():
            persona = "Persona: Tingkat Stres/Beban Paling Tinggi"
        elif avg_score == non_noise_means.min():
            persona = "Persona: Tingkat Stres/Beban Paling Rendah"
        else:
            persona = "Persona: Tingkat Stres/Beban Menengah"

        interpretations[cluster_id] = {
            "persona": persona,
            "description": description
        }
        
    return interpretations


def hasilkan_analisis_final(df, labels, kolom_numerik):
    """
    Menghasilkan semua output final (interpretasi, heatmap, dll.) 
    berdasarkan label clustering yang dipilih.
    """
    df['Cluster'] = labels
    
    # Hitung rata-rata keseluruhan sebelum dikelompokkan
    overall_means = df[kolom_numerik].mean()
    
    cluster_means = df.groupby('Cluster')[kolom_numerik].mean()
    interpretasi_klaster = generate_interpretations(cluster_means, overall_means)

    # Buat heatmap
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 7))
    # Abaikan noise dari heatmap jika ada
    sns.heatmap(cluster_means.drop(-1, errors='ignore').T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax_heatmap)
    ax_heatmap.set_title('Heatmap Karakteristik Setiap Klaster', fontsize=14, fontweight='bold')
    ax_heatmap.set_xlabel('Cluster', fontsize=12)
    ax_heatmap.set_ylabel('Pertanyaan Survei', fontsize=12)

    # Siapkan hasil untuk ditampilkan di Streamlit
    distribusi = df['Cluster'].value_counts().sort_index()
    
    return {
        "distribusi_klaster": distribusi,
        "rata_rata_klaster": cluster_means.round(2),
        "figur_heatmap": fig_heatmap,
        "dataframe_hasil": df,
        "interpretasi_klaster": interpretasi_klaster
    }