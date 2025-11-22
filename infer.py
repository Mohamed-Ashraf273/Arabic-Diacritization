import string

# here test your code
from preprocessing.arabic_preprocessing import ArabicPreprocessor
#  Training
with open("data/train.txt", "r", encoding="utf-8") as f:
    train = f.read()
cleaned_train= ArabicPreprocessor(train).apply()
print("Len of training before cleaning",len(train))
with open("data/cleaned_train.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_train)
print("Len of training after cleaning",len(cleaned_train))

# Validation
with open("data/val.txt", "r", encoding="utf-8") as f:
    valid = f.read()
print("Len of validation before cleaning",len(valid))
cleaned_valid= ArabicPreprocessor(valid).apply()
with open("data/cleaned_val.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_valid)
print("Len of validation after cleaning",len(cleaned_valid))
