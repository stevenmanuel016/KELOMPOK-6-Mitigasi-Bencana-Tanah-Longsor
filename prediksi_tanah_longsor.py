import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. LOAD DAN PREPROCESSING DATA ====================
print("="*60)
print("ANALISIS PREDIKSI LONGSOR SULAWESI TENGAH")
print("="*60)

# Load data
df = pd.read_csv('prediksi_longsor_sulawesi_tengah.csv')

# Tampilkan informasi dasar
print(f"\n📊 INFORMASI DATASET:")
print(f"Jumlah data: {len(df)} baris")
print(f"Jumlah fitur: {len(df.columns)} kolom")
print(f"\nStruktur data:")
print(df.info())
print(f"\nStatistik deskriptif:")
print(df.describe())

# ==================== 2. ANALISIS RISIKO ====================
print("\n" + "="*60)
print("📈 ANALISIS TINGKAT RISIKO")
print("="*60)

# Analisis distribusi risiko
risk_stats = df['risk_label'].value_counts()
print(f"\nDistribusi Label Risiko:")
print(risk_stats)

# Temukan wilayah dengan risiko tertinggi per tahun
print("\n🔍 WILAYAH DENGAN RISIKO TERTINGGI PER TAHUN:")
for year in df['year'].unique():
    year_data = df[df['year'] == year]
    max_risk = year_data.loc[year_data['risk_prob'].idxmax()]
    print(f"Tahun {year}: {max_risk['regency']} - Risk: {max_risk['risk_prob']:.3f}")

# ==================== 3. ANALISIS SPASIAL-TEMPORAL ====================
print("\n" + "="*60)
print("🗺️ ANALISIS SPASIAL-TEMPORAL")
print("="*60)

# Rata-rata risiko per kabupaten
avg_risk_regency = df.groupby('regency')['risk_prob'].agg(['mean', 'max', 'min']).sort_values('mean', ascending=False)
print("\n📊 RATA-RATA RISIKO PER KABUPATEN:")
print(avg_risk_regency.head(10))

# Trend risiko per tahun
yearly_risk = df.groupby('year')['risk_prob'].agg(['mean', 'max', 'std'])
print("\n📈 TREND RISIKO TAHUNAN:")
print(yearly_risk)

# ==================== 4. ANALISIS FAKTOR DOMINAN ====================
print("\n" + "="*60)
print("🔬 ANALISIS FAKTOR PENYEBAB")
print("="*60)

# Korelasi antar variabel numerik
numeric_cols = ['rainfall_mm', 'slope_deg', 'elevation_m', 'ndvi', 'population_density', 'previous_landslides', 'risk_prob']
correlation_matrix = df[numeric_cols].corr()

print("\n🔗 KORELASI DENGAN RISIKO LONGSOR:")
risk_corr = correlation_matrix['risk_prob'].sort_values(ascending=False)
for factor, corr in risk_corr.items():
    if factor != 'risk_prob':
        print(f"{factor:25} : {corr:+.3f}")

# ==================== 5. CLUSTERING WILAYAH ====================
print("\n" + "="*60)
print("🏙️  CLUSTERING WILAYAH BERDASARKAN KARAKTERISTIK")
print("="*60)

# Persiapan data untuk clustering
cluster_features = ['rainfall_mm', 'slope_deg', 'elevation_m', 'ndvi', 'risk_prob']
X = df[cluster_features]

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analisis cluster
cluster_analysis = df.groupby('cluster')[cluster_features].mean()
print("\n📊 KARAKTERISTIK SETIAP CLUSTER:")
print(cluster_analysis)

# ==================== 6. VISUALISASI ====================
print("\n" + "="*60)
print("📊 GENERATING VISUALISASI...")
print("="*60)

# Setup style
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ANALISIS KOMPREHENSIF PREDIKSI LONGSOR SULAWESI TENGAH', fontsize=16, fontweight='bold')

# 1. Distribusi Probabilitas Risiko
axes[0, 0].hist(df['risk_prob'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(df['risk_prob'].mean(), color='red', linestyle='--', linewidth=2, label=f'Rata-rata: {df["risk_prob"].mean():.3f}')
axes[0, 0].set_xlabel('Probabilitas Risiko')
axes[0, 0].set_ylabel('Frekuensi')
axes[0, 0].set_title('Distribusi Probabilitas Risiko Longsor')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Trend Risiko per Tahun
yearly_stats = df.groupby('year')['risk_prob'].agg(['mean', 'std'])
axes[0, 1].plot(yearly_stats.index, yearly_stats['mean'], marker='o', linewidth=2, markersize=8, color='green')
axes[0, 1].fill_between(yearly_stats.index, 
                       yearly_stats['mean'] - yearly_stats['std'], 
                       yearly_stats['mean'] + yearly_stats['std'], 
                       alpha=0.2, color='green')
axes[0, 1].set_xlabel('Tahun')
axes[0, 1].set_ylabel('Rata-rata Probabilitas Risiko')
axes[0, 1].set_title('Trend Risiko Longsor per Tahun')
axes[0, 1].grid(True, alpha=0.3)

# 3. Heatmap Korelasi
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1, 0])
axes[1, 0].set_title('Heatmap Korelasi Antar Variabel')

# 4. Scatter Plot: Curah Hujan vs Kemiringan Lereng
scatter = axes[1, 1].scatter(df['rainfall_mm'], df['slope_deg'], 
                            c=df['risk_prob'], cmap='Reds', 
                            s=df['population_density']/10, alpha=0.6)
axes[1, 1].set_xlabel('Curah Hujan (mm)')
axes[1, 1].set_ylabel('Kemiringan Lereng (derajat)')
axes[1, 1].set_title('Hubungan Curah Hujan, Kemiringan, dan Risiko')
plt.colorbar(scatter, ax=axes[1, 1], label='Probabilitas Risiko')

plt.tight_layout()
plt.savefig('analisis_longsor_sulawesi_tengah.png', dpi=300, bbox_inches='tight')
print("✅ Visualisasi disimpan sebagai 'analisis_longsor_sulawesi_tengah.png'")

# ==================== 7. REKOMENDASI ====================
print("\n" + "="*60)
print("🎯 REKOMENDASI MITIGASI")
print("="*60)

# Identifikasi wilayah prioritas
high_risk_threshold = df['risk_prob'].quantile(0.75)
priority_areas = df[df['risk_prob'] > high_risk_threshold]

print(f"\n🔴 WILAYAH PRIORITAS (Risiko > {high_risk_threshold:.3f}):")
for idx, row in priority_areas.iterrows():
    print(f"  • {row['regency']} ({row['year']}): Risk={row['risk_prob']:.3f}")

print("\n📋 REKOMENDASI UMUM:")
print("1. Pemantauan intensif di wilayah dengan curah hujan tinggi dan lereng curam")
print("2. Penghijauan dan konservasi tanah di area dengan NDVI rendah")
print("3. Sistem peringatan dini untuk permukiman padat penduduk")
print("4. Pembatasan pembangunan di zona rawan tinggi")
print("5. Edukasi masyarakat tentang mitigasi bencana")

# ==================== 8. EKSPOR HASIL ====================
print("\n" + "="*60)
print("EKSPOR HASIL ANALISIS")
print("="*60)

# Ekspor data dengan cluster
df_export = df.copy()
df_export['risk_category'] = pd.cut(df_export['risk_prob'], 
                                   bins=[0, 0.05, 0.07, 1], 
                                   labels=['Rendah', 'Sedang', 'Tinggi'])

df_export.to_csv('hasil_analisis_longsor_sulawesi_tengah.csv', index=False)
print("✅ Data hasil analisis disimpan sebagai 'hasil_analisis_longsor_sulawesi_tengah.csv'")

# Ringkasan akhir
print("\n" + "="*60)
print("RINGKASAN EKSEKUSI PROGRAM")
print("="*60)
print(f"✅ Data berhasil dianalisis: {len(df)} observasi")
print(f"✅ Rentang tahun: {df['year'].min()} - {df['year'].max()}")
print(f"✅ Jumlah kabupaten/kota: {df['regency'].nunique()}")
print(f"✅ Rata-rata probabilitas risiko: {df['risk_prob'].mean():.3f}")
print(f"✅ Wilayah dengan risiko tertinggi: {df.loc[df['risk_prob'].idxmax(), 'regency']}")
print(f"✅ Visualisasi disimpan: 'analisis_longsor_sulawesi_tengah.png'")
print(f"✅ Data hasil disimpan: 'hasil_analisis_longsor_sulawesi_tengah.csv'")

# Tampilkan visualisasi
plt.show()