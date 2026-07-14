import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline


print("\n" + "="*55)
print("[1] PENGATURAN DIREKTORI")
print("="*55)
# input
print("\nCONTOH PENULISAN ALAMAT DATASET: /home/pi/(nama folder proyek)/(nama folder data)")
while True:
    BASE_PATH = input("Masukkan alamat folder dataset: ").strip()
    if not os.path.isdir(BASE_PATH):
        print(f"-> [ERROR] Folder '{BASE_PATH}' tidak ditemukan! pastikan penulisan alamat sudah benar\n")
        continue
    break     
# output
print("\nCONTOH PENULISAN ALAMAT HASIL TRAINING: /home/pi/pelatihan/training")
while True:
    OUTPUT_FOLDER = input("Masukkan alamat folder output: ").strip()
    if OUTPUT_FOLDER:
        break

print("\n" + "="*55)
print("[2] PENGATURAN PARAMETER EKSTRAKSI DATA")
print("="*55)
# input batas baseline
while True:
    input_baseline = input("\nMasukkan batas akhir data baseline (contoh: 60): ").strip()
    if not input_baseline:
        continue
    try:
        batas_baseline = int(input_baseline)
        break
    except ValueError:
        print("\nINPUT TIDAK VALID")
# input batas sampel
while True:
    input_respon = input("\nMasukkan batas akhir data sampel (contoh: 300): ").strip()
    if not input_respon:
        continue
    try:
        batas_respon = int(input_respon)
        break
    except ValueError:
        print("\nINPUT TIDAK VALID")

SENSOR_NAMES = ['MQ-8', 'MQ-135', 'MQ-7', 'MQ-3', 'TGS-822', 'MQ-136', 'TGS-813', 'MQ-4']
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# EKSTRAKSI
def extract_features(df, b_end, r_end):
    data_gas = df[SENSOR_NAMES]
    
    # Baseline dan sampel
    baseline = data_gas.iloc[:b_end].mean()
    df_delta = data_gas - baseline
    df_respon = df_delta.iloc[b_end:r_end]
    
    # Statistik
    max_vals = df_respon.max().values
    mean_vals = df_respon.mean().values
    median_vals = df_respon.median().values
    std_vals = df_respon.std().values
    
    # Bentuk sinyal
    momentum = df_respon.diff().max().values  
    
    return np.concatenate([
        max_vals, mean_vals, median_vals, std_vals, 
        momentum
    ])


print("\n" + "="*55)
print("[3] PENGATURAN LABEL DATA")
print("="*55)
folders = sorted([f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))])

unique_prefixes = set()
for folder_name in folders:
    match = re.match(r"([a-zA-Z_]+)", folder_name)
    if match:
        unique_prefixes.add(match.group(1))

if not unique_prefixes:
    print("-> [ERROR] Format nama folder tidak dikenali! pastikan folder diawali dengan huruf (contoh: A1, B1, C1)")
    exit()

print(f"\nProgram mendeteksi {len(unique_prefixes)} kelompok data: {', '.join(unique_prefixes)}")
label_mapping = {}
for prefix in sorted(unique_prefixes):
    input_label = input(f"\nMasukkan nama kelas untuk awalan '{prefix}': ").strip()
    label_mapping[prefix] = input_label if input_label else prefix


print("\n" + "="*55)
print("[4] PEMBACAAN DAN EKSTRAKSI FITUR")
print("="*55)

X_list, y_list, file_names = [], [], []

for folder_name in folders:
    match = re.match(r"([a-zA-Z_]+)", folder_name)
    prefix = match.group(1) if match else "Unknown"
    
    label = label_mapping.get(prefix, "Unknown")
    f_path = os.path.join(BASE_PATH, folder_name)
    
    files = sorted([f for f in os.listdir(f_path) if f.endswith('.csv')])
    for file in files:
        try:
            df = pd.read_csv(os.path.join(f_path, file))
            if len(df) >= batas_respon:
                features = extract_features(df, batas_baseline, batas_respon)
                X_list.append(features)
                y_list.append(label)
                file_names.append(file)
        except Exception as e:
            print(f"-> Gagal memproses file {file}: {e}")

X = np.array(X_list)
y = np.array(y_list)

if len(X) == 0:
    print("\n[ERROR] Tidak ada data yang berhasil diekstrak. Cek kembali dataset Anda.")
    exit()

print(f"\nBerhasil mengekstrak {len(X)} sampel data")

# Buat kolom
headers = []
for s in SENSOR_NAMES: headers.append(f"{s}_Max")
for s in SENSOR_NAMES: headers.append(f"{s}_Mean")
for s in SENSOR_NAMES: headers.append(f"{s}_Median")
for s in SENSOR_NAMES: headers.append(f"{s}_Std")
for s in SENSOR_NAMES: headers.append(f"{s}_Momentum")

df_ekstraksi = pd.DataFrame(X, columns=headers)
df_ekstraksi.insert(0, 'File_Name', file_names)
df_ekstraksi.insert(1, 'Label_Kelas', y)
df_ekstraksi.to_csv(os.path.join(OUTPUT_FOLDER, 'data_ekstraksi_fitur.csv'), index=False)

# ENCODING DAN PEMBAGIAN DATA
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, os.path.join(OUTPUT_FOLDER, 'label_encoder.pkl'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# BUAT PIPELINE
pipeline = Pipeline([
    ('scaler', PowerTransformer()),
    ('rf', RandomForestClassifier(random_state=42))
])

# PARAMETER GRID
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [5, 10, None],
    'rf__min_samples_split': [2, 5],
    'rf__max_features': ['sqrt', 'log2']
}


print("\n" + "="*55)
print("[5] MEMBUAT MODEL")
print("="*55)
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rf_grid = GridSearchCV(pipeline, param_grid, cv=cv_strategy, n_jobs=-1)
rf_grid.fit(X_train, y_train)

# EVALUASI FINAL
best_model = rf_grid.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

print(f"\n KLASIFIKASI METRIK\n")
report = classification_report(y_test, y_pred, target_names=le.classes_)
filtered_report = '\n'.join([line for line in report.split('\n') if 'macro avg' not in line and 'weighted avg' not in line])
print(filtered_report)
joblib.dump(best_model, os.path.join(OUTPUT_FOLDER, 'model_pipeline.pkl'))


print("\n" + "="*55)
print("[6] MEMBUAT VISUALISASI")
print("="*55)
plt.figure(figsize=(8,6))
sns.heatmap(
    confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples', 
    xticklabels=le.classes_, yticklabels=le.classes_
)
plt.title(f'Confusion Matrix (Akurasi: {acc:.2f}%)')
plt.ylabel('AKTUAL')
plt.xlabel('PREDIKSI')
plt.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix.png'), dpi=300)
plt.close()

# VISUALISASI FEATURE IMPORTANCE
best_rf = best_model.named_steps['rf']
importances = best_rf.feature_importances_
feature_names = np.array(headers) 

top_indices = np.argsort(importances)[-25:][::-1]
plt.figure(figsize=(12, 6))
plt.title('Fitur Sensor Terbaik', fontsize=14, pad=15)
plt.bar(range(len(top_indices)), importances[top_indices], color='royalblue', align='center')
plt.xticks(range(len(top_indices)), feature_names[top_indices], rotation=45, ha='right')
plt.ylabel('Nilai Kepentingan Relatif')
plt.xlabel('Fitur Ekstraksi')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout() 
plt.savefig(os.path.join(OUTPUT_FOLDER, 'feature_importance.png'), dpi=300)
plt.close()

print(f"\nProses training berhasil dan tersimpan di: {OUTPUT_FOLDER}\n")
