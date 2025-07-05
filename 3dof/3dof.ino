/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 https://www.arduino.cc/en/Tutorial/LibraryExamples/Sweep
*/

#include <Servo.h>
#include <Arduino.h>
Servo myservo;
Servo myservo1; 
Servo myservo2; // create servo object to control a servo
// twelve servo objects can be created on most boards
#include<mwc_stepper.h>

#define EN_PIN 3
#define DIR_PIN 2
#define STEP_PIN 5

#define RPM 2
#define RPM1 2
#define ms1 4
#define ms2 6
#define ms3 7 
#define PULSE 2000  
//in case of using microstepping just multiply the steps by the microsteps 
int ClOCKWISE =0;
int OTHERWISE =1;

MWCSTEPPER nema23(EN_PIN, DIR_PIN, STEP_PIN);
int pos = 0;    // variable to store the servo position
int theta1=abs(61-90);
int theta2=58;
int theta3=38;

int steps=theta1/1.8;

int mt=2;

void setup() {
   Serial.begin(9600);  // Ensure the baud rate matches the Python script
  Serial.println("Ready to receive data...");
nema23.init();
myservo.attach(8);
myservo1.attach(11);
myservo2.attach(3);//gripper
digitalWrite(ms1,HIGH);


myservo.write(120);
myservo1.write(90);//link2
myservo2.write(0); //gripper

delay(3000);

   
              // tell servo to go to position in variable 'pos'
  // attaches the servo on pin 9 to the servo object
}
int th1=73;
int th2=38;
int th3=54;
int Theta1=78;
int Theta2=32;
int Theta3=79;
void loop() {

  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');  // Read data until newline
    Serial.print("Received: ");
    Serial.println(data);

    // Split the data into individual angles
    sscanf(data.c_str(), "%d,%d,%d,%d,%d,%d", &th1, &th2, &th3, &Theta1, &Theta2, &Theta3);
    Serial.print(th1); Serial.print(",");
    Serial.print(th2); Serial.print(",");
    Serial.print(th3); Serial.print(",");
    Serial.print(Theta1); Serial.print(",");
    Serial.print(Theta2); Serial.print(",");
    Serial.println(Theta3);
 
    move(th1,th2,th3,Theta1,Theta2,Theta3);
    
    // Print received values
    
  


}

}


void move(int t1,int t2,int t3,int T1,int T2,int T3)
{
  
  if(t1<=90)
{
ClOCKWISE=0;
OTHERWISE=1;
}
else
{
  ClOCKWISE=1;
  OTHERWISE=0;
} 
  delay(1000);
t1=abs(t1-90);
steps=mt*(abs(t1/1.8));

  nema23.set(OTHERWISE, RPM1, PULSE);

  for (size_t i = 0; i < steps; i++)
  {
    nema23.run();
  }
   delay(1000);


  delay(2000);
if(t3<=90){ 
 for (pos = 90; pos >= t3; pos -= 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo1.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  } 
  delay(1000);}

  else{
    for (pos = 90; pos <= t3; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo1.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  } 
  delay(1000);
  }    
for (pos = 130; pos >= t2; pos -= 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  }
     
  
  delay(2000);
  myservo2.write(170); 

  for (pos = t2; pos <= 130; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  }
  delay(1000);

  if(t3<=90){
  for (pos = t3; pos <= 90; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo1.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  } 
  }

  else{
    for (pos = t3; pos >= 90; pos -= 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo1.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  }
delay(2000);
  }
nema23.set(ClOCKWISE, RPM, PULSE);

  for (size_t i = 0; i < steps; i++)
  {
    nema23.run();
  }

  delay(1000);










delay(1000);
if(T1<=90)
{
ClOCKWISE=0;
OTHERWISE=1;
}
else
{
  ClOCKWISE=1;
  OTHERWISE=0;
}
T1=abs(T1-90);
int Steps=mt*(abs(T1/1.8));
 
  nema23.set(OTHERWISE, RPM1, PULSE);

  for (size_t i = 0; i < Steps; i++)
  {
    nema23.run();
  }
   delay(1000);


  delay(2000);
if(T3<=90){ 
 for (pos = 90; pos >= T3; pos -= 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo1.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  } 
  delay(1000);}

  else{
    for (pos = 90; pos <= T3; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo1.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  } 
  delay(1000);
  }    
for (pos = 130; pos >= T2; pos -= 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  }
     
  
  delay(2000);
  myservo2.write(0); 

  
  for (pos = T2; pos <= 130; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  }
  delay(1000);

   if(T3<=90){
  for (pos = T3; pos <= 90; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo1.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  } 
  }

  else{
    for (pos = T3; pos >= 90; pos -= 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo1.write(pos);              // tell servo to go to position in variable 'pos'
    delay(20);                       // waits 15 ms for the servo to reach the position
  }
delay(2000);
  }
nema23.set(ClOCKWISE, RPM, PULSE);

  for (size_t i = 0; i < Steps; i++)
  {
    nema23.run();
  }

  delay(1000);
}








  

