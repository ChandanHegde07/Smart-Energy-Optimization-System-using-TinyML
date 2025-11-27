import serial
import csv
import datetime
import time

SERIAL_PORT = 'COM3'  # Windows: 'COM3', Mac/Linux: '/dev/ttyUSB0' or '/dev/tty.usbmodem...'
BAUD_RATE = 9600
FILENAME = 'Sensor Data.csv'

# 1. Connect to Arduino
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting to serial port: {e}")
    exit()

try:
    with open(FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["date", "Temperature", "Light", "PIR"])
            print("Created new file with header.")
except Exception as e:
    print(f"Error opening file: {e}")
    exit()

print("Logging data... Press Ctrl+C to stop.")

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            if line:
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                sensor_values = line.split(',')
                
                if len(sensor_values) == 3: 
                    csv_row = [current_date] + sensor_values
                    
                    # Save to CSV
                    with open(FILENAME, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(csv_row)
                    
                    print(f"Logged: {csv_row}")
                    
except KeyboardInterrupt:
    print("\nLogging stopped.")
    ser.close()
