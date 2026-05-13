// Servo and buzzer controller for Dual Access Control System.
// Reads "GRANTED" or "DENIED" from the laptop over USB serial and
// activates the corresponding actuator.

#include <ESP32Servo.h>

const int SERVO_PIN  = 18;
const int BUZZER_PIN = 19;
const int BAUD_RATE  = 9600;

Servo accessServo;

void setup() {
    Serial.begin(BAUD_RATE);
    accessServo.attach(SERVO_PIN);
    pinMode(BUZZER_PIN, OUTPUT);
    accessServo.write(0);       // locked position
    digitalWrite(BUZZER_PIN, LOW);
}

void loop() {
    if (Serial.available() > 0) {
        String message = Serial.readStringUntil('\n');
        message.trim();

        if (message == "GRANTED") {
            accessServo.write(90);  // open position
            delay(3000);
            accessServo.write(0);   // return to locked
        } else if (message == "DENIED") {
            digitalWrite(BUZZER_PIN, HIGH);
            delay(1000);
            digitalWrite(BUZZER_PIN, LOW);
        }
    }
}
