import pyttsx3 as pt3

if __name__ == '__main__':
    engine = pt3.init()
    engine.setProperty('rate', 200)    # 设置语音速度
    engine.setProperty('volume', 1)   # 设置语音音量

    engine.say("I'm feeling happy today!")   # 将文本转换为语音

    engine.runAndWait()