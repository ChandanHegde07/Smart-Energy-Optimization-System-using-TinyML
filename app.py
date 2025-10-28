from flask import Flask, render_template, request, jsonify
from datetime import datetime
import numpy as np


app = Flask(__name__)


class DataStore:
    def __init__(self):
        self.timestamps = []
        self.predictions = []
        self.energy_saved = []
        self.pir_values = []
        self.ldr_values = []
        self.temperature_values = []  
        self.total_energy = 0.0
        self.lights_prevented = 0
        self.current_pir = 0
        self.current_ldr = 0
        self.current_temp = 0.0  
        self.current_prediction = "Waiting for data..."
        self.max_temp = 0.0  
        self.min_temp = 100.0  
    
    def add_data(self, timestamp, pir, ldr, temp, prediction, energy):  
        self.timestamps.append(timestamp)
        self.pir_values.append(pir)
        self.ldr_values.append(ldr)
        self.temperature_values.append(temp)  
        self.predictions.append(1 if prediction == 'Light ON' else 0)
        self.total_energy += energy
        self.energy_saved.append(round(self.total_energy, 3))
        
        if temp > self.max_temp:
            self.max_temp = temp
        if temp < self.min_temp:
            self.min_temp = temp
        
        if len(self.timestamps) > 20:
            self.timestamps.pop(0)
            self.pir_values.pop(0)
            self.ldr_values.pop(0)
            self.temperature_values.pop(0)  
            self.predictions.pop(0)
            self.energy_saved.pop(0)
    
    def update_current(self, pir, ldr, temp, prediction):  
        self.current_pir = pir
        self.current_ldr = ldr
        self.current_temp = temp  
        self.current_prediction = prediction


data_store = DataStore()


# --- TinyML Model ---
def load_tinyml_model():
    """Load your TinyML model here"""
    return None


model = load_tinyml_model()


def neural_network_predict(pir, ldr, temperature):  
    """
    Enhanced TinyML prediction logic with temperature consideration
    """
    if pir == 1 and ldr < 50:
        if temperature > 35:
            return 'Light OFF (Heat)'
        else:
            return 'Light ON'
    elif pir == 1 and ldr >= 50:
        return 'Light OFF (Bright)'
    else:
        return 'Light OFF (No Motion)'


def calculate_energy_savings(prediction, pir, ldr, temperature): 
    """Calculate energy saved by AI decision"""
    energy_per_second = (60 / 1000) / 3600
    measurement_interval = 5
    
    if ldr > 50 and 'OFF' in prediction and pir == 1:
        data_store.lights_prevented += 1
        return energy_per_second * measurement_interval
    
    if temperature > 35 and 'Heat' in prediction:
        data_store.lights_prevented += 1
        return energy_per_second * measurement_interval * 1.2
    
    if pir == 0 and 'OFF' in prediction:
        return energy_per_second * measurement_interval * 0.5
    
    return 0.0


# --- ROUTES ---

@app.route('/')
def home():
    """Home page with navigation"""
    return render_template('home.html')


@app.route('/dashboard')
def index():
    """Serve main dashboard (formerly index)"""
    return render_template('index.html')


@app.route('/classroom_simulation')
def classroom_simulation():
    """Serve classroom simulation page"""
    return render_template('classroom_simulation.html')


@app.route('/update', methods=['POST'])
def update_data():
    """
    Endpoint for ESP32 to send sensor data
    Expected JSON: {"pir": 0/1, "ldr": 0-100, "temperature": 20-40}
    """
    try:
        data = request.json
        pir_value = int(data.get('pir', 0))
        ldr_value = int(data.get('ldr', 0))
        temp_value = float(data.get('temperature', 25.0))  
        
        prediction = neural_network_predict(pir_value, ldr_value, temp_value)
        
        energy_saved = calculate_energy_savings(prediction, pir_value, ldr_value, temp_value)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        data_store.add_data(timestamp, pir_value, ldr_value, temp_value, prediction, energy_saved)
        data_store.update_current(pir_value, ldr_value, temp_value, prediction)
        
        print(f"[{timestamp}] PIR={pir_value}, LDR={ldr_value}, TEMP={temp_value}°C → {prediction} | Energy: {energy_saved:.6f} kWh")
        
        return jsonify({'success': True, 'prediction': prediction})
    
    except Exception as e:
        print(f"Error in /update: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/get_prediction')
def get_prediction():
    """Endpoint for frontend to fetch current data and statistics"""
    try:
        co2_saved = data_store.total_energy * 0.5
        cost_saved = data_store.total_energy * 0.12
        
        if len(data_store.temperature_values) > 0:
            avg_temp = sum(data_store.temperature_values) / len(data_store.temperature_values)
            max_temp = data_store.max_temp
            min_temp = data_store.min_temp
        else:
            avg_temp = 0.0
            max_temp = 0.0
            min_temp = 0.0
        
        response_data = {
            'pir': data_store.current_pir,
            'ldr': data_store.current_ldr,
            'temperature': round(data_store.current_temp, 1),  
            'prediction': data_store.current_prediction,
            'energy_saved': round(data_store.total_energy, 3),
            'lights_prevented': data_store.lights_prevented,
            'co2_saved': round(co2_saved, 2),
            'cost_saved': round(cost_saved, 2),
            'avg_temperature': round(avg_temp, 1),  
            'max_temperature': round(max_temp, 1),  
            'min_temperature': round(min_temp, 1),  
            'chart_data': {
                'timestamps': data_store.timestamps,
                'predictions': data_store.predictions,
                'energy_saved': data_store.energy_saved,
                'temperatures': data_store.temperature_values  
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in /get_prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/simulate', methods=['GET'])
def simulate_data():
    """Simulate sensor data for testing without hardware"""
    import random
    import time
    
    count = int(request.args.get('count', 1))
    results = []
    
    for _ in range(count):
        pir_value = random.choice([0, 0, 0, 1])
        ldr_value = random.randint(0, 100)
        temp_value = round(random.uniform(22.0, 34.0), 1)  
        
        prediction = neural_network_predict(pir_value, ldr_value, temp_value)
        energy_saved = calculate_energy_savings(prediction, pir_value, ldr_value, temp_value)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        data_store.add_data(timestamp, pir_value, ldr_value, temp_value, prediction, energy_saved)
        data_store.update_current(pir_value, ldr_value, temp_value, prediction)
        
        results.append({
            'timestamp': timestamp,
            'pir': pir_value,
            'ldr': ldr_value,
            'temperature': temp_value,
            'prediction': prediction,
            'energy_saved': round(energy_saved, 6)
        })
        
        if count > 1:
            time.sleep(0.1)
    
    return jsonify({
        'success': True,
        'simulated': True,
        'count': count,
        'data': results,
        'message': f'Generated {count} simulated sensor reading(s) with temperature'
    })


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Smart Energy TinyML Dashboard with Temperature Monitoring")
    print("=" * 70)
    print("📡 Server accessible at:")
    print("   - Home Page: http://127.0.0.1:5001")
    print("   - Dashboard: http://127.0.0.1:5001/dashboard")
    print("   - Classroom Simulation: http://127.0.0.1:5001/classroom_simulation")
    print("   - Network: http://YOUR_IP:5001")
    print("=" * 70)
    print("🧪 Test without hardware: http://127.0.0.1:5001/simulate?count=20")
    print("📊 Waiting for ESP32 sensor data...")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
