import numpy as np
import csv
import re
from typing import List, Optional

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Catalog:
    """
    represents a catalog (aka corpus) of works, used in both model training and prediction
    Modified for Hindi/Devanagari text support
    """

    def __init__(self, padding: str = 'pre', oov_token='<OOV>', language='hindi'):
        self.catalog_items: List[str] = []
        self.tokenizer = Tokenizer(oov_token=oov_token, char_level=False)
        self.max_sequence_length = 0
        self.total_words = 0
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self._padding = padding
        self.language = language

    def _preprocess_hindi_text(self, text: str) -> str:
        """
        Preprocess Hindi text for better tokenization
        
        :param text: raw Hindi text
        :return: preprocessed text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Unicode (important for Devanagari)
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Remove English characters if desired (optional)
        # text = re.sub(r'[a-zA-Z]', '', text)
        
        # Remove special characters but keep Devanagari range
        # Devanagari Unicode range: \u0900-\u097F
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        
        return text.strip()

    def add_file_to_catalog(self, file_name: str) -> None:
        """
        add a text file to the catalog
        Now supports Hindi text with UTF-8 encoding

        :param file_name: file name with lyrics/text
        :return: None
        """
        with open(file_name, 'r', encoding='utf-8') as text_file:
            for line in text_file:
                if self.language == 'hindi':
                    processed_line = self._preprocess_hindi_text(line)
                else:
                    processed_line = line.lower()
                
                if processed_line:  # Only add non-empty lines
                    self.catalog_items.append(processed_line)

    def add_csv_file_to_catalog(self, file_name: str, text_column: int, skip_first_line: bool = True,
                                delimiter: str = ',') -> None:
        """
        add a csv, tsv or other delimited file to the catalog
        Now supports Hindi text with UTF-8 encoding

        :param file_name: file name with lyrics/text
        :param text_column: column number to select, 0 based
        :param skip_first_line: skip first line of text
        :param delimiter: delimiter to use as separator
        :return: None
        """
        with open(file_name, 'r', encoding='utf-8') as text_file:
            csv_reader = csv.reader(text_file, delimiter=delimiter)
            if skip_first_line:
                next(csv_reader)
            for row in csv_reader:
                if len(row) > text_column:
                    if self.language == 'hindi':
                        processed_line = self._preprocess_hindi_text(row[text_column])
                    else:
                        processed_line = row[text_column].lower()
                    
                    if processed_line:
                        self.catalog_items.append(processed_line)

    def tokenize_catalog(self) -> None:
        """
        tokenize the contents of the catalog, and set properties accordingly (ex: total_words, labels)

        :return: None
        """
        # Filter out empty items
        self.catalog_items = [item for item in self.catalog_items if item.strip()]
        
        if not self.catalog_items:
            raise ValueError("No valid text in catalog after preprocessing")

        # tokenizer: fit, sequence, pad
        self.tokenizer.fit_on_texts(self.catalog_items)

        # create a list of n-gram sequences
        input_sequences = []

        for line in self.catalog_items:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        if not input_sequences:
            raise ValueError("No sequences generated. Check your input data.")

        # pad sequences
        self.max_sequence_length = max([len(item) for item in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_length,
                                                 padding=self._padding))

        self.features = input_sequences[:, :-1]
        labels_temp = input_sequences[:, -1]

        self.total_words = len(self.tokenizer.word_index) + 1
        self.labels = keras.utils.to_categorical(labels_temp, num_classes=self.total_words)

        print(f"Total vocabulary size: {self.total_words}")
        print(f"Maximum sequence length: {self.max_sequence_length}")
        print(f"Total training sequences: {len(input_sequences)}")

    def generate_lyrics_text(self, model: keras.Sequential, seed_text: str, word_count: int) -> str:
        """
        generate lyrics using the provided model and properties
        Now handles Hindi text properly

        :param model: model used to generate text
        :param seed_text: starter text (in Hindi if language='hindi')
        :param word_count: total number of words to return
        :return: starter text + generated text
        """
        if self.language == 'hindi':
            seed_text = self._preprocess_hindi_text(seed_text)
        
        seed_text_word_count = len(seed_text.split())
        words_to_generate = word_count - seed_text_word_count

        for _ in range(words_to_generate):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_length - 1, padding=self._padding)
            predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

            predicted_index = int(predicted)
            output_word = self.tokenizer.index_word.get(predicted_index)
            
            if output_word is not None and output_word != '<OOV>':
                seed_text += ' ' + output_word

        return seed_text