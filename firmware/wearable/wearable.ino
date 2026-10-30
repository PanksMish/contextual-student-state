// Wearable firmware: XIAO ESP32S3 + MAX30102 (PPG/SpO2) + Grove GSR + MAX30205 (skin temp) + MPU6050 (motion).
// Output: one JSON line per second on USB serial at 115200 baud, parsed by css/runtime/wearable_io.py:
//   {"t":ms,"hr":bpm,"spo2":pct,"gsr":uS,"temp":C,"ax":g,"ay":g,"az":g,"ir":raw}
// hr = 0 means no valid beat; the host treats low IR or hr = 0 as no finger contact.
//
// Libraries (Arduino Library Manager): SparkFun MAX3010x Pulse and Proximity Sensor Library,
// Adafruit MPU6050, Protocentral MAX30205. Board: "XIAO_ESP32S3".
// Wiring: all I2C devices on SDA = D4 (GPIO5), SCL = D5 (GPIO6), 3V3 and GND; Grove GSR signal on A0 (D0).
// NOT FLASHED OR TESTED by the authors of this repository file: validate against the oximeter and
// thermistor protocol of the paper (Supplement S4) before using any reading.
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "Protocentral_MAX30205.h"

MAX30105 ppg;
Adafruit_MPU6050 mpu;
MAX30205 tempSensor;

const int GSR_PIN = A0;
const uint8_t BUF = 100;                 // 4 s at 25 samples/s for the SpO2 algorithm
uint32_t irBuf[BUF], redBuf[BUF];
uint8_t idx = 0; bool filled = false;
int32_t spo2 = 0; int8_t spo2Valid = 0; int32_t hrAlg = 0; int8_t hrValid = 0;

// beat-to-beat heart rate (moving average of the last 4 intervals)
const byte RATES = 4; float rates[RATES]; byte rateSpot = 0; long lastBeat = 0; float bpm = 0; byte rateCount = 0;

unsigned long lastPrint = 0, lastSample = 0;

float gsrMicroSiemens() {
  long sum = 0;
  for (int i = 0; i < 10; i++) sum += analogRead(GSR_PIN);
  int raw = (sum / 10) >> 2;             // 12-bit ADC to the 10-bit scale of the Grove formula
  if (raw >= 511) raw = 510;
  float ohm = ((1024.0 + 2.0 * raw) * 10000.0) / (512.0 - raw);   // Grove GSR skin resistance
  return 1000000.0 / ohm;                // conductance in microsiemens
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  if (!ppg.begin(Wire, I2C_SPEED_FAST)) { Serial.println("{\"error\":\"max30102\"}"); while (1) delay(1000); }
  // brightness, averaging, mode (red+IR), sample rate, pulse width, ADC range
  ppg.setup(60, 4, 2, 100, 411, 4096);
  if (!mpu.begin()) { Serial.println("{\"error\":\"mpu6050\"}"); while (1) delay(1000); }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  tempSensor.begin();
  analogReadResolution(12);
}

void loop() {
  ppg.check();
  while (ppg.available()) {
    uint32_t ir = ppg.getFIFOIR(), red = ppg.getFIFORed();
    if (checkForBeat(ir)) {
      long dt = millis() - lastBeat; lastBeat = millis();
      float b = 60.0 / (dt / 1000.0);
      if (b > 40 && b < 180) {
        rates[rateSpot++ % RATES] = b; if (rateCount < RATES) rateCount++;
        float s = 0; for (byte i = 0; i < rateCount; i++) s += rates[i]; bpm = s / rateCount;
      }
    }
    irBuf[idx] = ir; redBuf[idx] = red; idx++;
    if (idx == BUF) { idx = 0; filled = true; maxim_heart_rate_and_oxygen_saturation(irBuf, BUF, redBuf, &spo2, &spo2Valid, &hrAlg, &hrValid); }
    ppg.nextSample();
  }
  if (millis() - lastPrint >= 1000) {
    lastPrint = millis();
    sensors_event_t a, g, t; mpu.getEvent(&a, &g, &t);
    uint32_t irNow = ppg.getIR();
    bool contact = irNow > 50000;
    if (!contact) { bpm = 0; rateCount = 0; }
    Serial.printf("{\"t\":%lu,\"hr\":%.1f,\"spo2\":%d,\"gsr\":%.3f,\"temp\":%.2f,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,\"ir\":%lu}\n",
                  millis(), contact ? bpm : 0.0f, (contact && spo2Valid && filled) ? (int)spo2 : 0, gsrMicroSiemens(), tempSensor.getTemperature(),
                  a.acceleration.x / 9.81, a.acceleration.y / 9.81, a.acceleration.z / 9.81, (unsigned long)irNow);
  }
}
