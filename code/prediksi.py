import pandas as pd
import numpy as np
import joblib
import time
import os
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


print("\n\nPREDIKSI REAL-TIME")
print("\n" + "="*55)
print("[1] PENGATURAN DIREKTORI MODEL")
print("="*55)
print("\nCONTOH PENULISAN ALAMAT: /home/pi/(nama folder proyek)/(nama folder hasil training)")
while True:
    FOLDER_MODEL = input("\nMasukkan alamat folder hasil training: ").strip()
    
    if not os.path.isdir(FOLDER_MODEL):
        print(f"ERROR--Folder '{FOLDER_MODEL}' tidak ditemukan\n")
        continue
        
    MODEL_PATH = os.path.join(FOLDER_MODEL, 'model_pipeline.pkl')
    ENCODER_PATH = os.path.join(FOLDER_MODEL, 'label_encoder.pkl')
    
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(ENCODER_PATH):
        print("ERROR--File model_pipeline.pkl atau label_encoder.pkl tidak ditemukan di folder tersebut\n")
        continue
        
    break


print("\n" + "="*55)
print("[2] PENGATURAN WAKTU PENGUJIAN")
print("="*55)
while True:
    input_baseline = input("\nMasukkan durasi baseline (contoh: 60): ").strip()
    try:
        batas_baseline = int(input_baseline) if input_baseline else 60
        break
    except ValueError:
        print("INPUT TIDAK VALID\n")

while True:
    input_respon = input("\nMasukkan durasi total pengujian (contoh: 300): ").strip()
    try:
        batas_respon = int(input_respon) if input_respon else 300
        break
    except ValueError:
        print("INPUT TIDAK VALID\n")

# INISIALISASI HARDWARE
i2c = busio.I2C(board.SCL, board.SDA)

ads1 = ADS.ADS1115(i2c, address=0x48)
ads2 = ADS.ADS1115(i2c, address=0x49)
ads1.gain = 2
ads2.gain = 2

SENSOR_MAPPING = {
    "MQ-8": (ads1, 0), "MQ-135": (ads1, 1), "MQ-7": (ads1, 2), "MQ-3": (ads1, 3),
    "TGS-822": (ads2, 0), "MQ-136": (ads2, 1), "TGS-813": (ads2, 2), "MQ-4": (ads2, 3)
}
SENSOR_NAMES = list(SENSOR_MAPPING.keys())

def read_adc_volt(adc_obj, adc_pin_number):
    channel = AnalogIn(adc_obj, adc_pin_number)
    return round(channel.voltage * 2.8, 4)


# EKSTRAKSI
def extract_realtime_features(df, b_end, r_end):
    data_gas = df[SENSOR_NAMES]
    
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
    
    return np.concatenate([max_vals, mean_vals, median_vals, std_vals, momentum])

# MESIN PREDIKSI
def jalankan_uji_realtime():
    pipeline_model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    print("\n" + "="*55)
    print("[3] PROSES PENGUJIAN")
    print("="*55)

    sampel_aktual = input("\nMasukkan nama sampel yang dideteksi: ").strip()
    if not sampel_aktual:
        sampel_aktual = "Tidak Diketahui"
    
    data_points = []
    
    print(f"\n[FASE 1] PENGAMBILAN DATA BASELINE ({batas_baseline} DETIK)")
    print("-> Pastikan chamber kosong dan bersih")
    
    for detik in range(1, batas_baseline + 1):
        readings = [read_adc_volt(obj, pin) for s, (obj, pin) in SENSOR_MAPPING.items()]
        data_points.append(readings)
        
        if detik % 10 == 0 or detik == batas_baseline:
            print(f" Detik ke-{detik:>3}/{batas_baseline} | TGS-822: {readings[4]:.2f} V | MQ-7: {readings[2]:.2f} V | MQ-3: {readings[3]:.2f} V")
        time.sleep(1)
        
    print(f"\n[FASE 2] MASUKKAN SAMPEL KE DALAM CHAMBER")
    time.sleep(3)
    
    sisa_waktu = batas_respon - batas_baseline
    print(f"-> Merekam respons sensor selama {sisa_waktu} detik (Detik {batas_baseline + 1} - {batas_respon})")
    
    for detik in range(batas_baseline + 1, batas_respon + 1):
        readings = [read_adc_volt(obj, pin) for s, (obj, pin) in SENSOR_MAPPING.items()]
        data_points.append(readings)
        
        if detik % 15 == 0 or detik == batas_respon:
            print(f" Detik ke-{detik:>3}/{batas_respon} | TGS-822: {readings[4]:.2f} V | MQ-7: {readings[2]:.2f} V | MQ-3: {readings[3]:.2f} V")
        time.sleep(1)
        
    df = pd.DataFrame(data_points, columns=SENSOR_NAMES) 
    data_gas = df[SENSOR_NAMES]
    baseline_cek = data_gas.iloc[:batas_baseline].mean()
    df_delta_cek = data_gas - baseline_cek
    df_respon_cek = df_delta_cek.iloc[batas_baseline:batas_respon]
    
    max_deltas = df_respon_cek.max()
    lonjakan_maksimal = max_deltas.max()
    BATAS_MINIMAL_VOLTASE = 0.15

    print("\n RESPONS SENSOR (ΔV Maksimal)")
    print(f"  MQ-8   : {max_deltas['MQ-8']:>6.3f} V   |   TGS-822 : {max_deltas['TGS-822']:>6.3f} V")
    print(f"  MQ-135 : {max_deltas['MQ-135']:>6.3f} V   |   MQ-136  : {max_deltas['MQ-136']:>6.3f} V")
    print(f"  MQ-7   : {max_deltas['MQ-7']:>6.3f} V   |   TGS-813 : {max_deltas['TGS-813']:>6.3f} V")
    print(f"  MQ-3   : {max_deltas['MQ-3']:>6.3f} V   |   MQ-4    : {max_deltas['MQ-4']:>6.3f} V")

    if lonjakan_maksimal < BATAS_MINIMAL_VOLTASE:
        print("\n" + "="*55)
        print("TIDAK ADA SAMPEL YANG DIDETEKSI")
        print("(Gas sensor tidak mendeteksi lonjakan voltase yang cukup)")
        print("="*55 + "\n")
    else:

        fitur_mentah = extract_realtime_features(df, batas_baseline, batas_respon).reshape(1, -1)
        
        kode_prediksi = pipeline_model.predict(fitur_mentah)[0]
        nama_kelas = encoder.inverse_transform([kode_prediksi])[0]
        probabilitas = np.max(pipeline_model.predict_proba(fitur_mentah)) * 100

        print("\n" + "="*55)
        print("HASIL UJI REAL-TIME".center(55, " "))
        print("="*55)
        print(f" SAMPEL AKTUAL : {sampel_aktual.upper()}")
        print(f" HASIL PREDIKSI: {nama_kelas.upper()}")
        print(f" PROBABILITAS  : {probabilitas:.2f} %")
        print("="*55 + "\n")

if __name__ == "__main__":
    try:
        jalankan_uji_realtime()
    except KeyboardInterrupt:
        print("\n\nPengujian dihentikan paksa oleh pengguna")