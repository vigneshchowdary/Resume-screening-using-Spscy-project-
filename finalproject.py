#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import pickle
import random


# In[2]:


train_data = pickle.load(open(r'C:\Users\Vignesh Chowdary\Downloads\Resume-and-CV-Summarization-and-Parsing-with-Spacy-in-Python-master (1)\Resume-and-CV-Summarization-and-Parsing-with-Spacy-in-Python-master/train_data.pkl', 'rb'))
train_data[21]


# In[3]:


nlp = spacy.blank('en')

def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last = True)
    
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])
            
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(15):
            print("Statring iteration " + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                except Exception as e:
                    pass
                
            print(losses)


# In[4]:


train_model(train_data)


# In[5]:


nlp.to_disk('nlp_model')


# In[6]:


nlp_model = spacy.load('nlp_model')


# In[7]:


train_data[0][0]


# In[8]:


import sys, fitz
from tkinter import *
import tkinter as tk
from tkinter.ttk import *
from tkinter.filedialog import askopenfile

root = Tk()
root.geometry('1980x1080')
v = tk.StringVar()
def open_file():
    global fw
    file = askopenfile(mode ='r', filetypes =[('PDF  Files', '*.pdf')])
    fw=file.name

def extract():
    fname = f"{fw}"
    doc = fitz.open(fname)
    text = ""
    for page in doc:
        text = text + str(page.getText())
    tx = " ".join(text.split('\n'))
    doc = nlp_model(tx)
    for ent in doc.ents:
         br=(f'{ent.label_.upper():{30}} - {ent.text}')         
         e = Entry(root,width=170,justify=LEFT)
         e.insert(END, br)   
         e.pack()
         print(br)
            

qw=Label(root,text="RESUME SCREENING PROJECT",font='Helvetica 45 bold')
qw.place(x=327,y=350)
btn = Button(root, text ='SELECT PDF', command = open_file)
btn.place(x=720,y=490)
btn = Button(root, text ='EXTRACT', command = extract)
btn.place(x=720,y=540)
root.configure(background="skyblue")
root.mainloop()


# In[30]:





# In[ ]:





# In[ ]:




