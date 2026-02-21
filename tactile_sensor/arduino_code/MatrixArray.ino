// 3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing
// Code by Binghao Huang
// Website: https://binghao-huang.github.io/3D-ViTac/

// Copyright (c) 2024 Binghao Huang
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#define BAUD_RATE                 2000000
// #define ROW_COUNT                 16
#define ROW_COUNT                 8
#define COLUMN_COUNT              16

#define PIN_ADC_INPUT             A0
#define PIN_SHIFT_REGISTER_DATA   2 // D2
#define PIN_SHIFT_REGISTER_CLOCK  3 // D3
// MUX是用于行选择的多路复用器
#define PIN_MUX_CHANNEL_0         4 // D4
#define PIN_MUX_CHANNEL_1         5 // D5
#define PIN_MUX_CHANNEL_2         6 // D6
#define PIN_MUX_CHANNEL_3         7 // D7
#define PIN_MUX_INHIBIT_0         8 // D8
// #define PIN_MUX_INHIBIT_1         9

// PORTD是端口D输出器，用于控制8个数字引脚位PD0~PD7（在arduino上为D0~D7）
// 数据线
#define SET_SR_DATA_HIGH()        PORTD|=B00000100
#define SET_SR_DATA_LOW()         PORTD&=~B00000100
// 时钟线
#define SET_SR_CLK_HIGH()         PORTD|=B00001000
#define SET_SR_CLK_LOW()          PORTD&=~B00001000


// #define ROWS_PER_MUX              16
#define ROWS_PER_MUX              8
#define MUX_COUNT                 1
// #define CHANNEL_PINS_PER_MUX      4
#define CHANNEL_PINS_PER_MUX      3


#ifndef cbi
#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))
#endif
#ifndef sbi
#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit))
#endif

int current_enabled_mux = MUX_COUNT - 1;  //init to number of last mux so enabled mux increments to first mux on first scan.

void setup()
{
  Serial.begin(BAUD_RATE);
  // 设置引脚模式，输入/输出
  pinMode(PIN_ADC_INPUT, INPUT);
  pinMode(PIN_SHIFT_REGISTER_DATA, OUTPUT);
  pinMode(PIN_SHIFT_REGISTER_CLOCK, OUTPUT);
  pinMode(PIN_MUX_CHANNEL_0, OUTPUT);
  pinMode(PIN_MUX_CHANNEL_1, OUTPUT);
  pinMode(PIN_MUX_CHANNEL_2, OUTPUT);
  pinMode(PIN_MUX_CHANNEL_3, OUTPUT);
  pinMode(PIN_MUX_INHIBIT_0, OUTPUT);
  // pinMode(PIN_MUX_INHIBIT_1, OUTPUT);

  // 设置分频器系数，它将时钟频率/分频系数=ADC采样频率
  // 分频系数越大，采样频率越慢，精度更好
  sbi(ADCSRA,ADPS2);  //set ADC prescaler to CLK/16
  cbi(ADCSRA,ADPS1);
  cbi(ADCSRA,ADPS0);
}

void loop()
{
  for(int i = 0; i < ROW_COUNT; i ++)
  {
    setRow(i);
    shiftColumn(true);
    shiftColumn(false);                    
    for(int j = 0; j < COLUMN_COUNT; j ++)
    {
      int raw_reading = analogRead(PIN_ADC_INPUT);
      byte send_reading = (byte) (lowByte(raw_reading >> 2));
      shiftColumn(false);
      Serial.print(send_reading);
      Serial.print(" ");
    }
    Serial.println();
  }
  Serial.println();
  // delay(20);
}


void setRow(int row_number)
{
  // 这里先拉高，后拉低的目的是什么呢？
  if((row_number % ROWS_PER_MUX) == 0) 
  {
    digitalWrite(PIN_MUX_INHIBIT_0 + current_enabled_mux, HIGH);  
    current_enabled_mux ++;
    if(current_enabled_mux >= MUX_COUNT)
    {
      current_enabled_mux = 0;
    }
    digitalWrite(PIN_MUX_INHIBIT_0 + current_enabled_mux, LOW); 
  }
  // 设置row_number对应的引脚的高低电平值
  for(int i = 0; i < CHANNEL_PINS_PER_MUX; i ++)
  {
    // 如果row_number的二进制的第i位为1，则对应引脚输出高电平
    if(bitRead(row_number, i))  // 等价于 ((row_number >> i) & 0x1)
    {
      digitalWrite(PIN_MUX_CHANNEL_0 + i, HIGH);
    }
    else
    {
      digitalWrite(PIN_MUX_CHANNEL_0 + i, LOW);
    }
  }
}

void shiftColumn(boolean is_first)
{
  if(is_first)
  {
    SET_SR_DATA_HIGH();
  }
  SET_SR_CLK_HIGH();
  SET_SR_CLK_LOW();
  if(is_first)
  {
    SET_SR_DATA_LOW();
  }
}

// void printFixed(byte value)
// {
//   Serial.print(value);
// }


