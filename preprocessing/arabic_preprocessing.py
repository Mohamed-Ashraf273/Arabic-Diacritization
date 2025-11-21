from preprocessing.preprocessor import Preprocessor
import pickle
import re
import string
from bs4 import BeautifulSoup


class ArabicPreprocessor(Preprocessor):
    def __init__(self, text):
        super().__init__(text)
        self.split_punct = {".", ",", "،", "؛", ":", "?", "؟", "«", "»"}

    def remove_html_tags(self):
        self.text = BeautifulSoup(self.text, "html.parser").get_text("")

    def remove_english_characters(self):
        self.text = re.sub(r"[A-Za-z]", "", self.text)

    def remove_numbers(self):
        self.text = re.sub(r"[0-9٠-٩]+", "", self.text)

    def remove_urls(self):
        # This remove any URL http, https, www ,domain names, path after domain name,
        url_pattern = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
        self.text = re.sub(url_pattern, "", self.text)

    def remove_kashida(self):
        self.text = re.sub(r"ـ+", "", self.text)

    def remove_extra_spaces(self):
        self.text = re.sub(r"[ \t]+", " ", self.text)
        # let the line breaks for splitting later
        self.text = "\n".join([line.strip() for line in self.text.split("\n")])

    def solve_diacritics_issues(self):
        with open("utils/diacritics.pickle", "rb") as f:
            diacritics_chars = "".join(list(pickle.load(f)))
        # using regex by removing the whitespaces for ‘Ending Diacritics
        self.text = re.sub(rf"\s+([{diacritics_chars}])", r"\1", self.text)
        # keeping only the first diacritic in ‘Multiple Consecutive Diacritics
        self.text = re.sub(f"({diacritics_chars})\\1+", r"\1", self.text)

    def remove_punctuation(self):
        unwanted_punct = "".join(set(string.punctuation) - set(self.split_punct))
        self.text = re.sub(f"[{re.escape(unwanted_punct)}]", "", self.text)

    def apply(self):
       pass
