// RFID reader for Dual Access Control System (Phase 3).
// Reads the UID from an RC522 RFID card and sends it to the laptop over USB serial.
// The laptop replies with "GRANTED" or "DENIED" to trigger the servo/buzzer controller.

// Placeholder — implement after Phase 2 is complete.
// Required library: MFRC522 by miguelbalboa (install via Arduino Library Manager).

// #include <SPI.h>
// #include <MFRC522.h>

// const int SS_PIN  = 5;
// const int RST_PIN = 22;
// MFRC522 rfid(SS_PIN, RST_PIN);

void setup() {
    Serial.begin(9600);
    // SPI.begin();
    // rfid.PCD_Init();
}

void loop() {
    // TODO: scan for card, read UID, send over Serial, wait for reply.
}
