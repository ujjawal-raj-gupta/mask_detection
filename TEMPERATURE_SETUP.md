# ESP32 + MLX90614 Temperature Setup

This project can include a person's measured temperature in the final webcam result.
The ESP32 reads the MLX90614 sensor and sends the temperature to Python over USB serial.

## Wiring

Use the ESP32 default I2C pins used in `esp32_mlx90614_temperature/esp32_mlx90614_temperature.ino`.

| MLX90614 Pin | ESP32 Dev Board Pin |
| --- | --- |
| VIN / VCC | 3V3 |
| GND | GND |
| SDA | GPIO 21 |
| SCL | GPIO 22 |

If your MLX90614 breakout specifically requires 5V VIN, follow that board's label, but keep I2C logic safe for the ESP32.

## Arduino IDE Setup

1. Install ESP32 board support in Arduino IDE.
2. Install these Arduino libraries:
   - `Adafruit MLX90614 Library`
   - `Adafruit BusIO`
3. Open `esp32_mlx90614_temperature/esp32_mlx90614_temperature.ino`.
4. Select your ESP32 Dev Board and upload the sketch.
5. Open Serial Monitor at `115200` baud. You should see lines like:

```text
READY:MLX90614
TEMP_C:36.42,AMBIENT_C:28.15
```

## Python Detector

Run the detector from the project root:

```powershell
.\.venv\Scripts\python.exe detect_mask_video.py
```

If auto-detect does not find the ESP32 serial port, pass it manually:

```powershell
.\.venv\Scripts\python.exe detect_mask_video.py --temp-port COM3
```

Replace `COM3` with the port shown in Arduino IDE.

## Final Result Logic

- Green signal: mask detected and temperature is below `37.5 C`.
- Red signal: no mask or temperature is `37.5 C` and above.
- Yellow/waiting signal: face or ESP32 temperature data is not available yet.

For reliable readings, keep the MLX90614 pointed at the person's forehead from a short, consistent distance.

## Web Dashboard

Run the browser dashboard instead of the OpenCV window:

```powershell
.\.venv\Scripts\python.exe dashboard_app.py --temp-port COM4
```

Then open:

```text
http://127.0.0.1:5000
```

The dashboard shows:
- live webcam frame with detection overlay
- mask status card
- temperature card
- final green/red/yellow signal
