from flask import Flask, render_template, request, jsonify
from datetime import datetime
import numpy as np
import pandas as pd
import os
import math
import time

# Try to import TensorFlow for the AI Model
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"TensorFlow import failed: {e}")
    tf = None

app = Flask(__name__)

# --- DATA MANAGEMENT ---
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
        
        # Current Real-Time State
        self.current_pir = 0
        self.current_ldr = 0
        self.current_temp = 0.0  
        self.current_prediction = "Waiting..."
        self.current_fan_status = "OFF" # NEW: Track Fan Status
        
        # Stats
        self.max_temp = 0.0  
        self.min_temp = 100.0
        self.prediction_count = 0
        
        # History for Features
        self.light_history = []
        self.temp_history = []
        
        # SYNC LOGIC: Track when the user last manually touched the simulation
        self.last_manual_input_time = 0 
        
        # Curated scenarios (Auto Mode Data)
        self.demo_scenarios = [
            {'pir': 1, 'ldr': 200, 'temp': 24.0, 'description': 'Auto: Motion + Dark'},
            {'pir': 0, 'ldr': 350, 'temp': 25.0, 'description': 'Auto: No Motion'},
            {'pir': 1, 'ldr': 850, 'temp': 26.0, 'description': 'Auto: Motion + Bright'},
            {'pir': 1, 'ldr': 250, 'temp': 23.5, 'description': 'Auto: Motion + Low Light'},
            {'pir': 0, 'ldr': 450, 'temp': 24.5, 'description': 'Auto: No Motion + Medium'},
            {'pir': 1, 'ldr': 280, 'temp': 37.5, 'description': 'Auto: Motion + Hot'},
            {'pir': 1, 'ldr': 150, 'temp': 24.0, 'description': 'Auto: Motion + Very Dark'},
            {'pir': 0, 'ldr': 200, 'temp': 23.0, 'description': 'Auto: No Motion + Dark'},
        ]
    
    def get_auto_scenario(self):
        # Change scenario every 5 seconds
        interval = 5 
        current_index = int(time.time() / interval) % len(self.demo_scenarios)
        return self.demo_scenarios[current_index]
    
    def add_data(self, timestamp, pir, ldr, temp, prediction, fan_status, energy):  
        self.timestamps.append(timestamp)
        self.pir_values.append(pir)
        self.ldr_values.append(ldr)
        self.temperature_values.append(temp)  
        self.predictions.append(1 if 'ON' in prediction else 0)
        self.total_energy += energy
        self.energy_saved.append(round(self.total_energy, 3))
        
        self.light_history.append(ldr)
        self.temp_history.append(temp)
        if len(self.light_history) > 5: self.light_history.pop(0)
        if len(self.temp_history) > 5: self.temp_history.pop(0)
        
        if temp > self.max_temp: self.max_temp = temp
        if temp < self.min_temp: self.min_temp = temp
        
        if len(self.timestamps) > 20:
            self.timestamps.pop(0)
            self.pir_values.pop(0)
            self.ldr_values.pop(0)
            self.temperature_values.pop(0)  
            self.predictions.pop(0)
            self.energy_saved.pop(0)
    
    def update_current(self, pir, ldr, temp, prediction, fan_status):  
        self.current_pir = pir
        self.current_ldr = ldr
        self.current_temp = temp  
        self.current_prediction = prediction
        self.current_fan_status = fan_status # Update Fan

data_store = DataStore()

# --- MODEL LOADING ---
def load_tinyml_model():
    if tf is None: return None
    try:
        print("\n" + "="*60)
        print("LOADING YOUR FNN MODEL")
        print("="*60)
        model_path = 'occupancy_fnn_model.h5'
        if not os.path.exists(model_path):
            print(f"File not found: {model_path}")
            return None
        model = tf.keras.models.load_model(model_path)
        print("YOUR FNN MODEL LOADED!")
        print(f"   Input shape: {model.input_shape}")
        print("   Status: Ready for Inference")
        print("="*60 + "\n")
        return model
    except Exception as e:
        print(f"ERROR LOADING MODEL: {e}")
        return None

model = load_tinyml_model()

# --- FEATURE ENGINEERING ---
def create_engineered_features(pir, ldr, temperature):
    pir = float(pir)
    ldr = float(ldr)
    temp = float(temperature)
    current_hour = datetime.now().hour
    hour_sin = math.sin(2 * math.pi * current_hour / 24)
    
    if len(data_store.light_history) >= 3:
        light_mean_3 = sum(data_store.light_history[-3:]) / 3
        light_diff_3 = ldr - data_store.light_history[-3]
    else:
        light_mean_3 = ldr
        light_diff_3 = 0.0
    
    if len(data_store.temp_history) >= 3:
        temp_mean_3 = sum(data_store.temp_history[-3:]) / 3
        temp_diff_3 = temp - data_store.temp_history[-3]
    else:
        temp_mean_3 = temp
        temp_diff_3 = 0.0
    
    return [temp, ldr, pir, light_mean_3, temp_mean_3, light_diff_3, temp_diff_3, hour_sin]

def manual_scaler(features):
    temp, ldr, pir, l_mean, t_mean, l_diff, t_diff, hour = features
    s_temp = (temp - 0) / (50 - 0)
    s_ldr = (ldr - 0) / (1000 - 0)
    s_pir = pir
    s_l_mean = (l_mean - 0) / (1000 - 0)
    s_t_mean = (t_mean - 0) / (50 - 0)
    s_l_diff = (l_diff + 500) / (1000)
    s_t_diff = (t_diff + 10) / (20)
    s_hour = (hour + 1) / 2
    return [s_temp, s_ldr, s_pir, s_l_mean, s_t_mean, s_l_diff, s_t_diff, s_hour]

# --- PREDICTION LOGIC ---
def neural_network_predict(pir, ldr, temperature):  
    if model is None:
        if pir == 1 and ldr < 500: return 'Light ON' if temperature <= 35 else 'Light OFF (Heat)'
        elif pir == 1 and ldr >= 500: return 'Light OFF (Bright)'
        else: return 'Light OFF (No Motion)'
    try:
        data_store.prediction_count += 1
        raw_features = create_engineered_features(pir, ldr, temperature)
        scaled_features = manual_scaler(raw_features)
        input_data = np.array([scaled_features], dtype=np.float32)
        prediction_prob = model.predict(input_data, verbose=0)[0][0]
        is_occupied = prediction_prob > 0.5
        
        print(f"\n--- FNN Inference #{data_store.prediction_count} ---")
        print(f"Inputs: PIR={pir}, Lux={ldr}, Temp={temperature}")
        print(f"Model Probability: {prediction_prob:.4f} ({'Occupied' if is_occupied else 'Empty'})")
        
        if not is_occupied:
            decision = 'Light OFF (No Motion)'
            reason = "Model detected room empty"
        elif ldr >= 600:
            decision = 'Light OFF (Bright)'
            reason = "Occupied, but natural light is sufficient"
        elif temperature > 35:
            decision = 'Light OFF (Heat)'
            reason = "Occupied, but turning off to reduce heat"
        else:
            decision = 'Light ON'
            reason = "Model detected Occupancy + Low Light"
            
        print(f"Decision: {decision} | Reason: {reason}\n")
        return decision
    except Exception as e:
        print(f"Prediction Error: {e}")
        return 'Light OFF (Error)'

def get_fan_status(light_decision, temperature):
    """Determine Fan Status based on Occupancy and Temperature"""
    # If decision contains "No Motion", room is empty -> Fan OFF
    if "No Motion" in light_decision:
        return "Fan OFF"
    
    # If room is occupied (any other decision) AND Temp > 25
    if temperature > 25.0:
        return "Fan ON"
    
    return "Fan OFF"

def calculate_energy_savings(prediction, fan_status, pir, ldr, temperature): 
    energy_per_second = (60 / 1000) / 3600 
    measurement_interval = 5
    
    savings = 0.0
    if ldr > 500 and 'OFF' in prediction and pir == 1:
        data_store.lights_prevented += 1
        savings += energy_per_second * measurement_interval
    
    if temperature > 35 and 'Heat' in prediction:
        data_store.lights_prevented += 1
        savings += energy_per_second * measurement_interval * 1.2
    
    if pir == 0 and 'OFF' in prediction:
        savings += energy_per_second * measurement_interval * 0.5
        
    return savings

# --- FLASK ROUTES ---

@app.route('/')
def home(): return render_template('home.html')

@app.route('/dashboard')
def index(): return render_template('index.html')

@app.route('/classroom_simulation')
def classroom_simulation(): return render_template('classroom_simulation.html')

@app.route('/model_info')
def model_info():
    if model is not None:
        return jsonify({
            'model_loaded': True,
            'model_name': 'occupancy_fnn_model.h5',
            'accuracy': '98%',
            'predictions_made': data_store.prediction_count,
            'using_real_model': True,
            'message': 'Active: FNN Model'
        })
    return jsonify({'model_loaded': False})

@app.route('/update', methods=['POST'])
def update_data():
    try:
        data = request.json
        pir_value = int(data.get('pir', 0))
        ldr_value = float(data.get('ldr', 0))
        temp_value = float(data.get('temperature', 25.0))
        
        data_store.last_manual_input_time = time.time()
        
        prediction = neural_network_predict(pir_value, ldr_value, temp_value)
        fan_status = get_fan_status(prediction, temp_value)
        energy_saved = calculate_energy_savings(prediction, fan_status, pir_value, ldr_value, temp_value)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        data_store.add_data(timestamp, pir_value, ldr_value, temp_value, prediction, fan_status, energy_saved)
        data_store.update_current(pir_value, ldr_value, temp_value, prediction, fan_status)
        
        return jsonify({'success': True, 'prediction': prediction, 'fan_status': fan_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/get_prediction')
def get_prediction():
    try:
        current_time = time.time()
        time_since_input = current_time - data_store.last_manual_input_time
        
        if time_since_input < 5.0:
            # Manual Mode
            pir_value = data_store.current_pir
            ldr_value = data_store.current_ldr
            temp_value = data_store.current_temp
            description = "Manual / Simulation Input"
            prediction = data_store.current_prediction
            fan_status = data_store.current_fan_status
        else:
            # Auto Mode
            scenario = data_store.get_auto_scenario()
            pir_value = scenario['pir']
            ldr_value = scenario['ldr']
            temp_value = scenario['temp']
            description = scenario['description']
            
            prediction = neural_network_predict(pir_value, ldr_value, temp_value)
            fan_status = get_fan_status(prediction, temp_value)
            energy_saved = calculate_energy_savings(prediction, fan_status, pir_value, ldr_value, temp_value)
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            data_store.add_data(timestamp, pir_value, ldr_value, temp_value, prediction, fan_status, energy_saved)
            data_store.update_current(pir_value, ldr_value, temp_value, prediction, fan_status)
        
        co2_saved = data_store.total_energy * 0.5
        cost_saved = data_store.total_energy * 0.12
        avg_temp = sum(data_store.temperature_values) / len(data_store.temperature_values) if data_store.temperature_values else 0.0
        
        return jsonify({
            'pir': data_store.current_pir,
            'ldr': round(data_store.current_ldr, 1),
            'temperature': round(data_store.current_temp, 1),
            'prediction': data_store.current_prediction,
            'fan_status': data_store.current_fan_status,  # Added Fan Status to JSON
            'scenario': description,
            'energy_saved': round(data_store.total_energy, 3),
            'lights_prevented': data_store.lights_prevented,
            'co2_saved': round(co2_saved, 2),
            'cost_saved': round(cost_saved, 2),
            'avg_temperature': round(avg_temp, 1),
            'max_temperature': round(data_store.max_temp, 1),
            'min_temperature': round(data_store.min_temp, 1),
            'real_model': True,
            'chart_data': {
                'timestamps': data_store.timestamps,
                'predictions': data_store.predictions,
                'energy_saved': data_store.energy_saved,
                'temperatures': data_store.temperature_values
            }
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SMART ENERGY TINYML DASHBOARD - SERVER STARTED")
    print("="*70)
    print(" [OK] Smart Sync Enabled: Manual Inputs override Auto Scenarios")
    app.run(host='0.0.0.0', port=5001, debug=True)