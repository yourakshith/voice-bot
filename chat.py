import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import pyttsx3
import speech_recognition as sr

r = sr.Recognizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Friday"
wake = bot_name.lower()
wakel = [wake,"Hey "+wake,"ok "+wake] #wakel = wake list
sleep = ["quit","terminate","break","abort","close","stop","finish"]

def talk(resp):
    engine=pyttsx3.init()
    voice = engine.getProperty('voices') 
    engine.setProperty('voice', voice[1].id)
    engine.say(resp)
    engine.runAndWait()


def listen():
    try:
        print(".",end="") #bot waiting for wake word
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source,phrase_time_limit = 3)
            key = r.recognize_google(audio)
            key = key.lower()
            # print(key)
            if key in wakel:
                print("!") #crct wake word
                return True
            elif key in sleep:
                print("*") #sleep word
                return False
            else: print("_",end="") #wrng word
            # return key
    except sr.RequestError as e:
        pass
        # print("~",end="") #Request Error
    except sr.UnknownValueError:
        pass
        # print("~",end="") #Unknown Value Error
    except KeyboardInterrupt:
        print("User stopped the bot, bye!")
        # return "quit"
        return False


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:#==1:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand... :("


if __name__ == "__main__":
    print(f"\n\nHello! Myself {bot_name}.")
    print(f"Let's talk! (say '{wake}' to wake, '{random.choice(sleep)}' to exit)\n")
    print("'.' -> listening \n'!' -> correct wake word \n'*' -> sleep word \n'_' -> some other word\n\n")
    while True:
        x=listen()
        if x:
            sentence = ""
            flag=False
            while sentence=="":
                try:
                    print("Listening...")
                    with sr.Microphone() as source:
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        audio = r.listen(source,phrase_time_limit = 5)
                        MyText = r.recognize_google(audio)
                        MyText = MyText.lower()
                        print("You: "+MyText)
                        sentence=MyText
                except sr.RequestError as e:
                    print("Could not request results; {0}".format(e))
                except sr.UnknownValueError:
                    print("unknown error occured") 
                except KeyboardInterrupt:
                    print("User stopped the bot, bye!")
                    flag=True
                    break
                except:
                    print("Something else happened!\nUse CTRL+C to stop the bot!")  
            if flag or sentence in sleep:
                print("User stopped the bot, bye!")
                break

            resp = get_response(sentence)
            print(f"{bot_name}: {resp}")
            talk(resp)
        elif x==None:
            pass
        elif not x:
            print("\nUser stopped the bot, bye!")
            break


# ** https://github.com/Rawkush/Chatbot/blob/master/intents.json
# https://github.com/Karan-Malik/Chatbot/blob/master/chatbot_codes/intents.json