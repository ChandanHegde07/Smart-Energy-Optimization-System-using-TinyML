#include <DHT.h>

#define PIR_PIN 7       
#define LDR_PIN A0       
#define DHT_PIN 2       
#define DHT_TYPE DHT11  

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(9600);

  pinMode(PIR_PIN, INPUT);
  dht.begin();
}

void loop() {

  float temperature = dht.readTemperature();  
  float humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Error reading DHT sensor!");
    return;
  }

  int lightLevel = analogRead(LDR_PIN);

  int pirState = digitalRead(PIR_PIN);

  Serial.print(temperature, 1);
  Serial.print(",");

  Serial.print(humidity, 1);
  Serial.print(",");

  Serial.print(lightLevel);
  Serial.print(",");

  Serial.println(pirState); 

  delay(1000);  
}
