import time
import datetime
import csv
import os
import board
import busio
import smbus2
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import pandas as pd
import matplotlib.pyplot as plt

# inisialisasi i2c
i2c = busio.I2C(board.SCL, board.SDA)
bus = smbus2.SMBus(1)

# inisialisasi ads1115
ads1 = ADS.ADS1115(i2c, address=0x48)
ads2 = ADS.ADS1115(i2c, address=0x49)
ads1.gain = 2
ads2.gain = 2

# insialiasi sensor gas {"nama sensor": (ads(n), (analog berapa? 0, 1, 2, 3)), sensor selanjutnya}
SENSOR_MAPPING = {
    "MQ-8": (ads1, 0), "MQ-135": (ads1, 1), "MQ-7": (ads1, 2), "MQ-3": (ads1, 3),
    "TGS-822": (ads2, 0), "MQ-136": (ads2, 1), "TGS-813": (ads2, 2), "MQ-4": (ads2, 3)
}

# pemrosesan data
def read_adc_volt(adc_obj, adc_pin_number):
    channel = AnalogIn(adc_obj, adc_pin_number)
    voltage = round(channel.voltage * 2.8, 4) 
    return voltage 

def save_to_csv(filename, data, header):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
            
        writer.writerows(data)

# pengambilan data
try:
    sample_name = input("\nNama sampel: ").strip()
    data = []
    
    # setup penyimpanan data
    BASE_DIR = "/home/pi/pelatihan/data/" # sesuialan dengan folder proyek anda dan beri nama untuk folder data
    SAMPLE_DIR = os.path.join(BASE_DIR, sample_name)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    filename = os.path.join(SAMPLE_DIR, f"{sample_name}.csv")
    
    input_total = input ("Total data : ")
    MAX_SAMPLES = int(input_total)
    sample_count = 0

    print(f"\nPengambilan data sampel '{sample_name}'")
    
    try:
        while sample_count < MAX_SAMPLES:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            sample_count += 1
            
            # baca sensor gas
            readings = {s: read_adc_volt(obj, pin) for s, (obj, pin) in SENSOR_MAPPING.items()}
            
            row = [timestamp] + [readings[s] for s in SENSOR_MAPPING]
            data.append(row)
            
            if sample_count % 1 == 0:
                print(f"Progres: {sample_count}/{MAX_SAMPLES} | TGS822: {readings['TGS-822']:.3f} V | MQ-3: {readings['MQ-3']:.3f} V | MQ-7: {readings['MQ-7']:.3f} V | TGS813: {readings['TGS-813']:.3f} V | MQ-4: {readings['MQ-4']:.3f} V | MQ-136: {readings['MQ-136']:.3f} V | MQ-135: {readings['MQ-135']:.3f} V | MQ-8: {readings['MQ-8']:.3f} V")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nBerhenti manual")
        
    finally:
        if data:
            # simpan data (csv)
            headers = ["Timestamp"] + list(SENSOR_MAPPING.keys())
            save_to_csv(filename, data, headers)
            print(f"\nData tersimpan di: {filename}")
            
            # buat grafik
            print("\nBerhasil membuat grafik")
            try:
                df = pd.read_csv(filename)
            
                df['Second'] = range(len(df))

                # plot gas
                plt.figure(figsize=(12, 6))
                for s in SENSOR_MAPPING.keys():
                    plt.plot(df['Second'], df[s], label=s)
                
                plt.title(f'Gas Sensor Array - {sample_name}')
                plt.xlabel('Waktu (s)')
                plt.ylabel('Tegangan (V)')
                plt.legend(loc='upper left', bbox_to_anchor=(1,1))
                plt.grid(True)
                plt.savefig(os.path.join(SAMPLE_DIR, f"{sample_name}_grafik.png"))
                plt.close()

            except Exception as e:
                print(f"Gagal grafik: {e}")

except Exception as e:
    print(f"Kesalahan: {e}")
finally:
    print("\nSeluruh proses pengambilan data berhasil\n")