#include <Adafruit_NeoPixel.h>
#include <Stepper.h>

//define NeoPixel Pin and Number of LEDs
#define LEDPIN 31
#define NUM_LEDS 12
#define STEPS 2038 // the number of steps in one revolution of your motor (28BYJ-48)
#define BUTTON_PIN 33

//create a NeoPixel strip
Adafruit_NeoPixel ring = Adafruit_NeoPixel(NUM_LEDS, LEDPIN, NEO_GRB + NEO_KHZ800);

Stepper stepper(STEPS, 30, 34, 32, 36);
Stepper stepper1(STEPS, 22, 26, 24, 28);
Stepper stepper_y(STEPS, 40, 44, 42, 46);

int potState = 0; 
int pot1State = 0;
int potyState = 0;

bool oldState = HIGH;
int showType = 0;

void setup() {
  // start the strip and blank it out
  Serial.begin(9600);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
 
  ring.begin();
  ring.show();

}

void loop() {
  // Get current button state.
  bool newState = digitalRead(BUTTON_PIN);

  // Check if state changed from high to low (button press).
  if (newState == LOW && oldState == HIGH) {
    // Short delay to debounce button.
    delay(20);
    // Check if button is still low after debounce.
    newState = digitalRead(BUTTON_PIN);
    if (newState == LOW) {
      showType++;
      if (showType > 1)
        showType=0;
      startShow(showType);
    }
  }

  // Set the last button state to the old state.
  oldState = newState;
  

  potState = analogRead(A0); //reads the values from the potentiometers
  pot1State = analogRead(A1); //
  potyState = analogRead(A2);
  
  Serial.println(pot1State); // sends joystick data to serial port for debuging
  stepper.setSpeed(5);
  stepper1.setSpeed(5);
  stepper_y.setSpeed(3);

  if (potState > 600){  //all code below controls movement
    stepper.step(10);
  }
  
  if (potState < 400){
    stepper.step(-10);
  }

  if (pot1State < 400){
    stepper1.step(10);
  }
  
  if (pot1State > 600){
    stepper1.step(-10);
  }
  
  if (potyState < 400){
    stepper_y.step(-10);
  }
  
  if (potyState > 600){
    stepper_y.step(10);
  }
}

void startShow(int i) {
  switch(i){
    case 0: ring.setPixelColor(0, 0, 0, 0);
            ring.setPixelColor(1, 0, 0, 0);
            ring.setPixelColor(2, 0, 0, 0);
            ring.setPixelColor(3, 0, 0, 0);
            ring.setPixelColor(4, 0, 0, 0);
            ring.setPixelColor(5, 0, 0, 0);
            ring.setPixelColor(6, 0, 0, 0);
            ring.setPixelColor(7, 0, 0, 0);
            ring.setPixelColor(8, 0, 0, 0);
            ring.setPixelColor(9, 0, 0, 0);
            ring.setPixelColor(10, 0, 0, 0);
            ring.setPixelColor(11, 0, 0, 0);
            ring.show();
            break;
    case 1: ring.setPixelColor(0, 200, 225, 175);
            ring.setPixelColor(1, 200, 225, 175);
            ring.setPixelColor(2, 200, 225, 175);
            ring.setPixelColor(3, 200, 225, 175);
            ring.setPixelColor(4, 200, 225, 175);
            ring.setPixelColor(5, 200, 225, 175);
            ring.setPixelColor(6, 200, 225, 175);
            ring.setPixelColor(7, 200, 225, 175);
            ring.setPixelColor(8, 200, 225, 175);
            ring.setPixelColor(9, 200, 225, 175);
            ring.setPixelColor(10, 200, 225, 175);
            ring.setPixelColor(11, 200, 225, 175);

            ring.show();
            break;
  }
}
