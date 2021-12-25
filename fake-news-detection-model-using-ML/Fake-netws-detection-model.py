import string
import re
import os
import numpy as np
import pandas as pd

from io import StringIO
from html.parser import HTMLParser

# Read input data
INPUT_FOLDER = "raw_datasets"
OUTPUT_FOLDER = "clean_datasets"

raw_fake = pd.read_csv('~/Desktop/fake.csv') 
raw_true = pd.read_csv('~/Desktop/test.csv') 
"""
Rework input
leave only text field and create a label column as input is composed of 2 datasets
1 = Fake, 0 = True
"""
train_fake = pd.DataFrame()
train_true = pd.DataFrame()

train_fake["text"] = raw_fake["text"]
train_fake["label"] = 1

train_true["text"] = raw_true["text"]
train_true["label"] = 0

train_clean = pd.concat([train_true, train_fake])

#Text preprocessing
class MLStripper(HTMLParser):
 def __init__(self):
        super().__init__()
 self.reset()
 self.strict = False
 self.convert_charrefs = True
 self.text = StringIO()

 def handle_data(self, d):
  self.text.write(d)

 def get_data(self):
  return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(str(html))
    return s.get_data()

def remove_tabulations(text):
    text = str(text)
    return(text.replace("\r", ' ').replace("\t", ' ').replace("\n", ' '))

def clean_text(text):
 # Remove HTML tags
    text = strip_tags(text)
 # Remove tabulation
    text = remove_tabulations(text)
 # convert to lower case
    text = text.lower()
 # Remove special characters
    text = re.sub('\[.*?\]', ' ', text)
 # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
 # normalize whitespace
    text = ' '.join(text.split())
    return text

def clean_text_basic(text):
 # remove whitespace before and after
    text = text.strip()
 # normalize whitespace
    text = ' '.join(text.split())
    return text
 
train_clean["text"]= train_clean["text"].apply(lambda x : clean_text(x))


output = train_clean[["text", "label"]]

output.to_csv(os.path.join(OUTPUT_FOLDER, "train_clean.csv"), index=False)