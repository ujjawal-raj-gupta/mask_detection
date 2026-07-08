#include <Wire.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

const unsigned long READ_INTERVAL_MS = 500;
unsigned long lastReadAt = 0;
bool sensorOk = false;

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("BOOT:MLX90614_SKETCH");

  Wire.begin(21, 22);
  Wire.setTimeOut(1000);
  Serial.println("STATUS:CHECKING_SENSOR");

  sensorOk = mlx.begin();
  if (!sensorOk) {
    Serial.println("ERROR:MLX90614_NOT_FOUND");
  } else {
    Serial.println("READY:MLX90614");
  }
}

void loop() {
  if (!sensorOk) {
    Serial.println("ERROR:MLX90614_NOT_FOUND");
    delay(2000);
    return;
  }

  if (millis() - lastReadAt < READ_INTERVAL_MS) {
    return;
  }

  lastReadAt = millis();

  float objectTempC = mlx.readObjectTempC();
  float ambientTempC = mlx.readAmbientTempC();

  if (isnan(objectTempC)) {
    Serial.println("ERROR:INVALID_TEMP_READING");
    return;
  }

  Serial.print("TEMP_C:");
  Serial.print(objectTempC, 2);
  Serial.print(",AMBIENT_C:");
  Serial.println(ambientTempC, 2);
}
