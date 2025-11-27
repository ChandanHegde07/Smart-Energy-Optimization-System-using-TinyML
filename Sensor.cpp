#include <DHT.h>

#define PIR_PIN 7        // PIR sensor digital pin
#define LDR_PIN A0       // LDR analog pin
#define DHT_PIN 2        // DHT data pin
#define DHT_TYPE DHT11   // Change to DHT22 if you use that sensor

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(9600);

  pinMode(PIR_PIN, INPUT);
  dht.begin();
}

void loop() {

  // --- Read Temperature & Humidity ---
  float temperature = dht.readTemperature();  // Celsius
  float humidity = dht.readHumidity();

  // Handle sensor read error
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Error reading DHT sensor!");
    return;
  }

  // --- Read LDR light level ---
  int lightLevel = analogRead(LDR_PIN);

  // --- Read PIR motion state ---
  int pirState = digitalRead(PIR_PIN);

  // --- Send data to serial in CSV format ---
  Serial.print(temperature, 1);
  Serial.print(",");

  Serial.print(humidity, 1);
  Serial.print(",");

  Serial.print(lightLevel);
  Serial.print(",");

  Serial.println(pirState); // newline = end of packet

  delay(1000);  
}
