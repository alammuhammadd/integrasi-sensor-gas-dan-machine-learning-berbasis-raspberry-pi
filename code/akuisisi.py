import time
import datetime
import csv
import os
import board
import busio
import smbus2
import pandas as pd
import matplotlib.pyplot as plt
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
try:
    import bme280
except ImportError:
    bme280 = None

# inisialisasi i2c
i2c = busio.I2C(board.SCL, board.SDA)
bus = smbus2.SMBus(1)

# inisialisasi ads1115
ads1 = ADS.ADS1115(i2c, address=0x48)
ads2 = ADS.ADS1115(i2c, address=0x49)
ads1.gain = 2
ads2.gain = 2

# inisialiasi sensor gas 
SENSOR_MAPPING = {
    "MQ-8": (ads1, 0), "MQ-135": (ads1, 1), "MQ-7": (ads1, 2), "MQ-3": (ads1, 3),
    "TGS-822": (ads2, 0), "MQ-136": (ads2, 1), "TGS-813": (ads2, 2), "MQ-4": (ads2, 3)
}

# pemrosesan data
def read_adc_volt(adc_obj, adc_pin_number):
    channel = AnalogIn(adc_obj, adc_pin_number)
    voltage = round(channel.voltage * 2.8, 4) 
    return voltage 

def read_bme280():
    bme = bme280.sample(bus, BME_ADDR)
    return round(bme.temperature, 2), round(bme.pressure, 2), round(bme.humidity, 2)

def save_to_csv(filename, data, header):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(data)
        
# pengaturan sensor lingkungan
tanya_bme = input("\nApakah ingin mengambil data dari sensor lingkungan? (y/n): ").strip().lower()
gunakan_bme = (tanya_bme == 'y')

if gunakan_bme:
    if bme280 is None:
        print("\nTidak dapat ambil data sensor lingkungan, library 'bme280' belum terinstall. Install terlebih dahulu 'pip install RPi.bme280'")
        gunakan_bme = False
    else:
        try:
            BME_ADDR = 0x76 
            bme280.load_calibration_params(bus, BME_ADDR)
            print("\nSensor BME berhasil diinisialisasi")
        except Exception as e:
            print(f"Gagal terhubung ke sensor BME: {e}. Mode lingkungan dinonaktifkan")
            gunakan_bme = False

try:
    sample_name = input("\nNama sampel: ").strip()
    data = []
    
    # setup penyimpanan data
    BASE_DIR = "/home/pi/pelatihan/data/"
    SAMPLE_DIR = os.path.join(BASE_DIR, sample_name)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    filename = os.path.join(SAMPLE_DIR, f"{sample_name}.csv")
    
    while True:
        input_total = input("Total data (contoh: 300): ").strip()
        try:
            MAX_SAMPLES = int(input_total)
            break
        except ValueError:
            print("-> Input harus berupa angka!")
            
    sample_count = 0

    print(f"\nMemulai pengambilan data sampel '{sample_name}'...")
    
    try:
        while sample_count < MAX_SAMPLES:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            sample_count += 1
            
            # 1. Baca sensor gas
            readings = {s: read_adc_volt(obj, pin) for s, (obj, pin) in SENSOR_MAPPING.items()}
            row = [timestamp] + [readings[s] for s in SENSOR_MAPPING]
            
            # 2. Baca sensor lingkungan (jika aktif)
            if gunakan_bme:
                t, p, h = read_bme280()
                row.extend([t, p, h])
                
                if sample_count % 1 == 0:
                    print(f"Progres: {sample_count}/{MAX_SAMPLES} | MQ-3: {readings['MQ-3']:.3f} V | MQ-7: {readings['MQ-7']:.3f} V | TGS822: {readings['TGS-822']:.3f} V || Suhu: {t:.1f} C | RH: {h:.1f} %")
            else:
                if sample_count % 1 == 0:
                    print(f"Progres: {sample_count}/{MAX_SAMPLES} | TGS822: {readings['TGS-822']:.3f} V | MQ-3: {readings['MQ-3']:.3f} V | MQ-7: {readings['MQ-7']:.3f} V | TGS813: {readings['TGS-813']:.3f} V | MQ-4: {readings['MQ-4']:.3f} V | MQ-136: {readings['MQ-136']:.3f} V | MQ-135: {readings['MQ-135']:.3f} V | MQ-8: {readings['MQ-8']:.3f} V")
            
            data.append(row)
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nBerhenti manual oleh pengguna")
        
    finally:
        if data:
            # Mengatur header sesuai pilihan mode
            headers = ["Timestamp"] + list(SENSOR_MAPPING.keys())
            if gunakan_bme:
                headers.extend(["Temp (C)", "Press (hPa)", "Humid (%)"])
                
            # Simpan ke CSV
            save_to_csv(filename, data, headers)
            print(f"\nData tersimpan di: {filename}")

            try:
                df = pd.read_csv(filename)
                df['Second'] = range(len(df))

                # 1. Grafik sensor gas
                plt.figure(figsize=(12, 6))
                for s in SENSOR_MAPPING.keys():
                    plt.plot(df['Second'], df[s], label=s)
                
                plt.title(f'Gas Sensor Array - {sample_name}')
                plt.xlabel('Waktu (s)')
                plt.ylabel('Tegangan (V)')
                plt.legend(loc='upper left', bbox_to_anchor=(1,1))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(SAMPLE_DIR, f"{sample_name}_gas.png"))
                plt.close()
                
                # 2. Grafik sensor lingkungan (jika BME aktif)
                if gunakan_bme:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    cols = ["Temp (C)", "Humid (%)", "Press (hPa)"]
                    colors = ['red', 'blue', 'green']
                    
                    for i, col in enumerate(cols):
                        axes[i].plot(df['Second'], df[col], color=colors[i])
                        axes[i].set_title(col)
                        axes[i].set_xlabel('Second')
                        axes[i].grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(SAMPLE_DIR, f"{sample_name}_env.png"))
                    plt.close()

                print("\nGrafik berhasil dibuat dan disimpan\n")
                
            except Exception as e:
                print(f"Gagal membuat grafik: {e}")

except Exception as e:
    print(f"\nKesalahan Fatal: {e}")
finally:
    pass