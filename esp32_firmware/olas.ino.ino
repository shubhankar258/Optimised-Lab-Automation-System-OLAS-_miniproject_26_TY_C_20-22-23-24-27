#include <EEPROM.h>
#include "RMaker.h"
#include "WiFi.h"
#include "WiFiProv.h"
#include <AceButton.h>
using namespace ace_button;

// ======== CONFIGURABLE FLAGS ==========
#define ENABLE_EEPROM true   // Enable EEPROM memory restore (true = Active)
#define USE_LATCHED_SWITCH true  // true = latched switch, false = push button
#define EEPROM_SIZE 10

// ======== DEVICE SETTINGS =============
const char *service_name = "PROV_12345";
const char *pop = "1234567";

char deviceName_1[] = "Switch1";
char deviceName_2[] = "Switch2";
char deviceName_3[] = "Switch3";
char deviceName_4[] = "Switch4";

// GPIO Setup
static uint8_t RelayPin1 = 23;
static uint8_t RelayPin2 = 19;
static uint8_t RelayPin3 = 18;
static uint8_t RelayPin4 = 26;

static uint8_t SwitchPin1 = 13;
static uint8_t SwitchPin2 = 12;
static uint8_t SwitchPin3 = 14;
static uint8_t SwitchPin4 = 27;

static uint8_t wifiLed = 2;
static uint8_t gpio_reset = 0;

bool toggleState_1 = LOW;
bool toggleState_2 = LOW;
bool toggleState_3 = LOW;
bool toggleState_4 = LOW;

ButtonConfig config1;
AceButton button1(&config1);
ButtonConfig config2;
AceButton button2(&config2);
ButtonConfig config3;
AceButton button3(&config3);
ButtonConfig config4;
AceButton button4(&config4);

static Switch my_switch1(deviceName_1, &RelayPin1);
static Switch my_switch2(deviceName_2, &RelayPin2);
static Switch my_switch3(deviceName_3, &RelayPin3);
static Switch my_switch4(deviceName_4, &RelayPin4);

void writeEEPROM(int addr, bool state) {
  if (ENABLE_EEPROM) {
    EEPROM.write(addr, state);
    EEPROM.commit();
    Serial.printf("EEPROM saved: addr %d = %d\n", addr, state);
  }
}

bool readEEPROM(int addr) {
  if (ENABLE_EEPROM) {
    return EEPROM.read(addr);
  }
  return false;
}

void setRelay(uint8_t pin, int addr, bool state) {
  digitalWrite(pin, !state); // active-low
  if (ENABLE_EEPROM) writeEEPROM(addr, state);
}

void buttonHandler(AceButton* button, uint8_t eventType, uint8_t, uint8_t relayPin, int eepromAddr, Switch &sw, bool &state) {
  bool newState = false;

  if (USE_LATCHED_SWITCH) {
    newState = (eventType == AceButton::kEventPressed);
  } else {
    if (eventType != AceButton::kEventReleased) return;
    newState = !(digitalRead(relayPin) == LOW);
  }

  setRelay(relayPin, eepromAddr, newState);
  state = newState;
  sw.updateAndReportParam(ESP_RMAKER_DEF_POWER_NAME, state);
  Serial.printf("Relay on pin %d toggled manually to %d\n", relayPin, state);
}

void setup() {
  Serial.begin(115200);

  if (ENABLE_EEPROM) EEPROM.begin(EEPROM_SIZE);

  toggleState_1 = ENABLE_EEPROM ? readEEPROM(0) : LOW;
  toggleState_2 = ENABLE_EEPROM ? readEEPROM(1) : LOW;
  toggleState_3 = ENABLE_EEPROM ? readEEPROM(2) : LOW;
  toggleState_4 = ENABLE_EEPROM ? readEEPROM(3) : LOW;

  pinMode(RelayPin1, OUTPUT); pinMode(RelayPin2, OUTPUT);
  pinMode(RelayPin3, OUTPUT); pinMode(RelayPin4, OUTPUT);
  pinMode(wifiLed, OUTPUT);

  pinMode(SwitchPin1, INPUT_PULLUP); pinMode(SwitchPin2, INPUT_PULLUP);
  pinMode(SwitchPin3, INPUT_PULLUP); pinMode(SwitchPin4, INPUT_PULLUP);
  pinMode(gpio_reset, INPUT);

  setRelay(RelayPin1, 0, toggleState_1);
  setRelay(RelayPin2, 1, toggleState_2);
  setRelay(RelayPin3, 2, toggleState_3);
  setRelay(RelayPin4, 3, toggleState_4);
  digitalWrite(wifiLed, LOW);

  config1.setEventHandler([](AceButton* b, uint8_t e, uint8_t s) {
    buttonHandler(b, e, s, RelayPin1, 0, my_switch1, toggleState_1);
  });
  config2.setEventHandler([](AceButton* b, uint8_t e, uint8_t s) {
    buttonHandler(b, e, s, RelayPin2, 1, my_switch2, toggleState_2);
  });
  config3.setEventHandler([](AceButton* b, uint8_t e, uint8_t s) {
    buttonHandler(b, e, s, RelayPin3, 2, my_switch3, toggleState_3);
  });
  config4.setEventHandler([](AceButton* b, uint8_t e, uint8_t s) {
    buttonHandler(b, e, s, RelayPin4, 3, my_switch4, toggleState_4);
  });

  button1.init(SwitchPin1);
  button2.init(SwitchPin2);
  button3.init(SwitchPin3);
  button4.init(SwitchPin4);

  Node my_node = RMaker.initNode("ESP32_Relay_4");
  my_switch1.addCb(write_callback);
  my_switch2.addCb(write_callback);
  my_switch3.addCb(write_callback);
  my_switch4.addCb(write_callback);

  my_node.addDevice(my_switch1);
  my_node.addDevice(my_switch2);
  my_node.addDevice(my_switch3);
  my_node.addDevice(my_switch4);

  RMaker.enableOTA(OTA_USING_PARAMS);
  RMaker.enableTZService();
  RMaker.enableSchedule();

  RMaker.start();
  WiFi.onEvent(sysProvEvent);
  WiFiProv.beginProvision(WIFI_PROV_SCHEME_BLE, WIFI_PROV_SCHEME_HANDLER_FREE_BTDM, WIFI_PROV_SECURITY_1, pop, service_name);

  my_switch1.updateAndReportParam(ESP_RMAKER_DEF_POWER_NAME, toggleState_1);
  my_switch2.updateAndReportParam(ESP_RMAKER_DEF_POWER_NAME, toggleState_2);
  my_switch3.updateAndReportParam(ESP_RMAKER_DEF_POWER_NAME, toggleState_3);
  my_switch4.updateAndReportParam(ESP_RMAKER_DEF_POWER_NAME, toggleState_4);

  Serial.println("Setup completed with EEPROM and mode flags.");
}

void loop() {
  if (digitalRead(gpio_reset) == LOW) {
    delay(100);
    int startTime = millis();
    while (digitalRead(gpio_reset) == LOW) delay(50);
    int duration = millis() - startTime;
    if (duration > 10000) {
      Serial.println("Factory reset triggered.");
      RMakerFactoryReset(2);
    } else if (duration > 3000) {
      Serial.println("WiFi reset triggered.");
      RMakerWiFiReset(2);
    }
  }

  digitalWrite(wifiLed, WiFi.status() == WL_CONNECTED);

  button1.check();
  button2.check();
  button3.check();
  button4.check();
}

void write_callback(Device *device, Param *param, const param_val_t val, void *priv_data, write_ctx_t *ctx) {
  const char *device_name = device->getDeviceName();
  const char *param_name = param->getParamName();

  if(strcmp(param_name, "Power") == 0) {
    bool newState = val.val.b;
    if (strcmp(device_name, deviceName_1) == 0) {
      setRelay(RelayPin1, 0, newState); toggleState_1 = newState; my_switch1.updateAndReportParam(param_name, newState);
    } else if (strcmp(device_name, deviceName_2) == 0) {
      setRelay(RelayPin2, 1, newState); toggleState_2 = newState; my_switch2.updateAndReportParam(param_name, newState);
    } else if (strcmp(device_name, deviceName_3) == 0) {
      setRelay(RelayPin3, 2, newState); toggleState_3 = newState; my_switch3.updateAndReportParam(param_name, newState);
    } else if (strcmp(device_name, deviceName_4) == 0) {
      setRelay(RelayPin4, 3, newState); toggleState_4 = newState; my_switch4.updateAndReportParam(param_name, newState);
    }
    Serial.printf("Write callback for %s: new state = %d\n", device_name, newState);
  }
}

void sysProvEvent(arduino_event_t *sys_event) {
  switch (sys_event->event_id) {
    case ARDUINO_EVENT_PROV_START:
      Serial.printf("Provisioning started: %s\n", service_name);
      printQR(service_name, pop, "ble");
      break;
    case ARDUINO_EVENT_WIFI_STA_CONNECTED:
      Serial.println("Connected to Wi-Fi");
      digitalWrite(wifiLed, true);
      break;
  }
}
