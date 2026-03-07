import requests
import random
import time
from datetime import datetime

SERVER_URL = "http://127.0.0.1:5001/update"  

def generate_realistic_data():
    """Generate realistic sensor data with temperature"""
    pir = random.choice([0, 0, 0, 1])
    ldr = random.randint(0, 100)
    
    if random.random() < 0.7:
        temperature = round(random.uniform(22.0, 28.0), 1)
    else:
        temperature = round(random.uniform(28.0, 35.0), 1)
    
    return {
        "pir": pir,
        "ldr": ldr,
        "temperature": temperature
    }

def main():
    print("=" * 70)
    print("🧪 TinyML Dashboard Simulator with Temperature Sensor")
    print("=" * 70)
    print(f"📡 Sending data to: {SERVER_URL}")
    print("🌡️  Temperature range: 22-35°C")
    print("💡 Light range: 0-100")
    print("👁️  Motion: Random detection")
    print("=" * 70)
    print("Press Ctrl+C to stop\n")
    
    counter = 0
    
    try:
        while True:
            counter += 1
            data = generate_realistic_data()
            
            try:
                response = requests.post(SERVER_URL, json=data, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    print(f"[{timestamp}] #{counter:03d} PIR={data['pir']} | "
                          f"LDR={data['ldr']:3d} | TEMP={data['temperature']:5.1f}°C | "
                          f"→ {result.get('prediction', 'N/A')}")
                else:
                    print(f"Error: HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"Cannot connect. Is Flask running on port 5001?")
                time.sleep(5)
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print(f"Simulation stopped after {counter} readings")
        print("=" * 70)

if __name__ == "__main__":
    main()
