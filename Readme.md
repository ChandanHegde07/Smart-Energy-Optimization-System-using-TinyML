# Smart Energy Optimization System using TinyML

A real-time IoT dashboard for intelligent lighting control using TinyML, ESP32, and environmental sensors. The system uses machine learning to make smart decisions about lighting based on motion detection, ambient light levels, and room temperature, resulting in significant energy savings.

---

## Features

- **Real-time Sensor Monitoring**: Live visualization of PIR motion, LDR light levels, and temperature data
- **TinyML Integration**: Intelligent decision-making using machine learning models
- **Energy Analytics**: Track energy savings, CO₂ reduction, and cost savings
- **Interactive Charts**: Beautiful Chart.js visualizations with historical data trends
- **Modern UI**: Glassmorphism design with smooth animations and responsive layout
- **Auto-refresh**: Configurable live data updates every 2 seconds
- **Simulation Mode**: Test without hardware using realistic sensor data generation

---

## Dashboard Components

### Sensor Monitoring
- **PIR Motion Sensor**: Detects human presence (0/1)
- **LDR Light Sensor**: Measures ambient brightness (0-100)
- **Temperature Sensor**: Monitors room temperature (°C)

### Analytics & Statistics
- Cumulative energy saved (kWh)
- CO₂ emissions reduced (kg)
- Cost savings (USD)
- Number of unnecessary light activations prevented
- Temperature statistics (min/max/average)

### Visualizations
- Energy savings trend chart
- Temperature monitoring graph
- AI decision history
- Real-time status indicators

---

## Technology Stack

**Backend:**
- Python 3.8+
- Flask (Web Framework)

**Frontend:**
- HTML5 / CSS3
- JavaScript (ES6+)
- Chart.js 4.4.0

**Hardware:**
- ESP32 Microcontroller
- DHT11/DHT22 Temperature Sensor
- PIR Motion Sensor (HC-SR501)
- LDR Light Dependent Resistor

**Libraries:**
- ArduinoJson
- DHT Sensor Library
- NumPy 
- Pandas
- Scikit-learn
- TensorFlow
- Seaborn

