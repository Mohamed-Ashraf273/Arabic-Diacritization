import string

# here test your code
from preprocessing.arabic_preprocessing import ArabicPreprocessor
text = "هذا نص عربي للاختبار"
with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(string.punctuation)