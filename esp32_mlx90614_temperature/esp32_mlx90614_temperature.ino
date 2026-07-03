#include <Wire.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

const unsigned long READ_INTERVAL_MS = 500;
unsigned long lastReadAt = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);  // ESP32 default I2C pins: SDA=21, SCL=22

  if (!mlx.begin()) {
    Serial.println("ERROR:MLX90614_NOT_FOUND");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("READY:MLX90614");
}

void loop() {
  if (millis() - lastReadAt < READ_INTERVAL_MS) {
    return;
  }

  lastReadAt = millis();

  float objectTempC = mlx.readObjectTempC();
  float ambientTempC = mlx.readAmbientTempC();

  Serial.print("TEMP_C:");
  Serial.print(objectTempC, 2);
  Serial.print(",AMBIENT_C:");
  Serial.println(ambientTempC, 2);
}
