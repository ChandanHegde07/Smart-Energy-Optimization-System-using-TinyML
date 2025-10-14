const REFRESH_INTERVAL = 2000;
let refreshTimer = null;
let energyChart = null;
let predictionChart = null;
let temperatureChart = null;

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const pirValue = document.getElementById('pir-value');
const ldrValue = document.getElementById('ldr-value');
const tempValue = document.getElementById('temp-value');
const resultDiv = document.getElementById('result');
const energySaved = document.getElementById('energySaved');
const co2Saved = document.getElementById('co2Saved');
const costSaved = document.getElementById('costSaved');
const lightsPrevented = document.getElementById('lightsPrevented');
const avgTemp = document.getElementById('avgTemp');
const maxTemp = document.getElementById('maxTemp');
const minTemp = document.getElementById('minTemp');
const autoRefreshToggle = document.getElementById('autoRefreshToggle');
const manualRefreshBtn = document.getElementById('manualRefresh');

function initCharts() {
    const energyCtx = document.getElementById('energyChart').getContext('2d');
    energyChart = new Chart(energyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Cumulative Energy Saved (kWh)',
                data: [],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: '#10b981',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { 
                        color: '#fff', 
                        font: { size: 13, weight: '600' }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#10b981',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `Energy Saved: ${context.parsed.y.toFixed(3)} kWh`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { 
                        color: '#fff',
                        font: { size: 11 },
                        callback: function(value) {
                            return value.toFixed(3);
                        }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: { 
                        color: '#fff',
                        font: { size: 10 },
                        maxRotation: 45, 
                        minRotation: 45 
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });

    const tempCtx = document.getElementById('temperatureChart').getContext('2d');
    temperatureChart = new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Room Temperature (°C)',
                data: [],
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.2)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: '#ff6b6b',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { 
                        color: '#fff', 
                        font: { size: 13, weight: '600' }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#ff6b6b',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `Temperature: ${context.parsed.y.toFixed(1)}°C`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 15,  
                    max: 40,  
                    ticks: { 
                        color: '#fff',
                        font: { size: 11 },
                        stepSize: 5,
                        callback: function(value) {
                            return value + '°C';
                        }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: { 
                        color: '#fff',
                        font: { size: 10 },
                        maxRotation: 45, 
                        minRotation: 45 
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });

    const predictionCtx = document.getElementById('predictionChart').getContext('2d');
    predictionChart = new Chart(predictionCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Light Decision (1=ON, 0=OFF)',
                data: [],
                backgroundColor: 'rgba(99, 102, 241, 0.7)',
                borderColor: '#6366f1',
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { 
                        color: '#fff', 
                        font: { size: 13, weight: '600' }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y === 1 ? 'Light ON' : 'Light OFF';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: { 
                        color: '#fff',
                        font: { size: 11 },
                        stepSize: 1,
                        callback: function(value) {
                            return value === 1 ? 'ON' : 'OFF';
                        }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: { 
                        color: '#fff',
                        font: { size: 10 },
                        maxRotation: 45, 
                        minRotation: 45 
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });
}

function updateStatus(connected) {
    if (connected) {
        statusDot.classList.remove('disconnected');
        statusText.textContent = 'Connected';
    } else {
        statusDot.classList.add('disconnected');
        statusText.textContent = 'Disconnected';
    }
}

async function updateDashboard() {
    try {
        const response = await fetch('/get_prediction');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        console.log('📊 Data received:', {
            pir: data.pir,
            ldr: data.ldr,
            temp: data.temperature,
            temp_data: data.chart_data?.temperatures
        });
        
        if (pirValue.textContent !== String(data.pir)) {
            pirValue.textContent = data.pir;
            pirValue.classList.add('updated');
            setTimeout(() => pirValue.classList.remove('updated'), 600);
        }
        
        if (ldrValue.textContent !== String(data.ldr)) {
            ldrValue.textContent = data.ldr;
            ldrValue.classList.add('updated');
            setTimeout(() => ldrValue.classList.remove('updated'), 600);
        }
        
        if (tempValue.textContent !== String(data.temperature)) {
            tempValue.textContent = data.temperature;
            tempValue.classList.add('updated');
            setTimeout(() => tempValue.classList.remove('updated'), 600);
        }
        
        energySaved.textContent = data.energy_saved.toFixed(3);
        co2Saved.textContent = data.co2_saved.toFixed(2);
        costSaved.textContent = `$${data.cost_saved.toFixed(2)}`;
        lightsPrevented.textContent = data.lights_prevented;
        
        avgTemp.textContent = `${data.avg_temperature}°C`;
        maxTemp.textContent = `${data.max_temperature}°C`;
        minTemp.textContent = `${data.min_temperature}°C`;
        
        resultDiv.textContent = data.prediction;
        
        if (data.chart_data && data.chart_data.timestamps.length > 0) {
            energyChart.data.labels = data.chart_data.timestamps;
            energyChart.data.datasets[0].data = data.chart_data.energy_saved;
            energyChart.update('none');
            
            temperatureChart.data.labels = data.chart_data.timestamps;
            temperatureChart.data.datasets[0].data = data.chart_data.temperatures;
            temperatureChart.update('none');
            
            console.log('🌡️ Temperature data updated:', data.chart_data.temperatures);
            
            predictionChart.data.labels = data.chart_data.timestamps;
            predictionChart.data.datasets[0].data = data.chart_data.predictions;
            predictionChart.update('none');
        }
        
        updateStatus(true);
        
    } catch (error) {
        console.error('❌ Dashboard update error:', error);
        updateStatus(false);
        resultDiv.textContent = 'Connection Error';
    }
}

function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(updateDashboard, REFRESH_INTERVAL);
}

function stopAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
}

autoRefreshToggle.addEventListener('change', function() {
    if (this.checked) {
        startAutoRefresh();
        updateDashboard();
    } else {
        stopAutoRefresh();
    }
});

manualRefreshBtn.addEventListener('click', function() {
    this.style.transform = 'rotate(360deg)';
    this.style.transition = 'transform 0.6s ease';
    setTimeout(() => this.style.transform = 'rotate(0deg)', 600);
    updateDashboard();
});

window.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Smart Energy Dashboard...');
    initCharts();
    updateDashboard();
    startAutoRefresh();
    console.log('✅ Dashboard with Temperature Monitoring initialized successfully!');
});
