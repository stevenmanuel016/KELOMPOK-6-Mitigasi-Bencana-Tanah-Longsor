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

