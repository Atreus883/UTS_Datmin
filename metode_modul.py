import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def manual_imputer(df, strategy='median'):
    """
    Fungsi untuk menangani nilai yang hilang (missing values) secara manual.
    Pengganti SimpleImputer.
    """
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() > 0:
            if strategy == 'median':
                fill_value = df_imputed[col].median()
            elif strategy == 'mean':
                fill_value = df_imputed[col].mean()
            else:
                fill_value = df_imputed[col].median()
            df_imputed[col].fillna(fill_value, inplace=True)
    return df_imputed

def manual_scaler(data):
    """
    Fungsi untuk melakukan standardisasi data secara manual (pengganti StandardScaler).
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - mean) / std

class ManualKMeans:
    """
    Kelas untuk implementasi algoritma K-Means secara manual (pengganti sklearn.cluster.KMeans).
    """
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = None

    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            for point_idx, point in enumerate(X):
                distances = [self._euclidean_distance(point, centroid) for centroid in self.cluster_centers_]
                closest_centroid_idx = np.argmin(distances)
                clusters[closest_centroid_idx].append(point)

            old_centroids = self.cluster_centers_.copy()

            for cluster_idx, cluster_points in enumerate(clusters):
                if cluster_points:
                    self.cluster_centers_[cluster_idx] = np.mean(cluster_points, axis=0)

            if np.all(self.cluster_centers_ == old_centroids):
                break
        
        self._calculate_inertia(X)
        return self

    def predict(self, X):
        labels = []
        for point in X:
            distances = [self._euclidean_distance(point, centroid) for centroid in self.cluster_centers_]
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

def generate_interpretations(cluster_means, overall_means):
    """
    Fungsi untuk menghasilkan teks interpretasi untuk setiap klaster.
    (Fungsi ini tidak perlu diubah)
    """
    interpretations = {}
    for cluster_id, means in cluster_means.iterrows():
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
    
    overall_means = df_numerik_imputed.mean()

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
    
    K_optimal = 3
    kmeans_final = ManualKMeans(n_clusters=K_optimal, random_state=42)
    cluster_labels = kmeans_final.fit_predict(data_scaled)
    df['Cluster'] = cluster_labels

    cluster_means = df.groupby('Cluster')[kolom_numerik].mean()
    interpretasi_klaster = generate_interpretations(cluster_means, overall_means)

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 7))
    sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax_heatmap)
    ax_heatmap.set_title('Heatmap Karakteristik Setiap Klaster', fontsize=14, fontweight='bold')
    ax_heatmap.set_xlabel('Cluster', fontsize=12)
    ax_heatmap.set_ylabel('Pertanyaan Survei', fontsize=12)

    pca = ManualPCA(n_components=2)
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
        "interpretasi_klaster": interpretasi_klaster
    }