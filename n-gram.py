import re
from textblob import TextBlob
from numpy.random import choice
from collections import defaultdict


def main() -> None:
    bigram = Bigram("cleaned_merged_fairy_tales_without_eos.txt")

    user_input = input("Enter text: ")
    n = int(input("How many words to generate: "))
    generated_text = bigram.generate_n_words(user_input.split()[-1], n)
    
    print(f"{user_input} {' '.join(generated_text)}")

class Bigram:
    """
    Reads in text file as corpus. Methods: generate_n_words based on input word and bigrams
    """
    def __init__(self, file: str) -> None:
        self.corpus = self._read_in_corpus(file)
        self.words = self._extract_words_from_corpus()
        self.bigrams = self._calculate_probabilities()

    def _read_corpus(self, file: str):
        with open(file) as f:
            return f.read()

    def _extract_words_from_corpus(self) -> list[str]:
        """Eextracts the words from corpus"""
        corpus_cleaned = re.sub('[^a-zA-Z\s]', '', self.corpus).lower()

        blob = TextBlob(corpus_cleaned)
        return blob.words

    def _get_bigrams(self) -> dict[dict[str]]:
        """Returns the bigrams and their frequencies"""
        # if the key doesn't exist, create a new dictionary with default integer values (0)
        bigrams = defaultdict(lambda: defaultdict(int))

        for word_1, word_2 in zip(self.words[:-1], self.words[1:]):
            bigrams[word_1][word_2] += 1

        return bigrams

    def _get_total_frequencies(self, bigrams: dict) -> dict:
        """Get total counts of all bigrams where the first word is the same"""
        total_frequencies = {}
        for first_word, first_word_dict in bigrams.items():
            total_frequency = sum([frequency for frequency in first_word_dict.values()])
            total_frequencies[first_word] = total_frequency
        return total_frequencies

    def _calculate_probabilities(self) -> dict:
        """Returns bigrams and their probabilities"""
        bigrams = self._get_bigrams()
        total_frequencies = self._get_total_frequencies(bigrams)

        for first_word, first_word_dict in bigrams.items():
            for second_word, frequency in first_word_dict.items():
                first_word_dict[second_word] = frequency/total_frequencies[first_word]
        return bigrams

    def _generate_next_word(self, word: str):
        """Returns the next word"""
        return choice(
            [second_word for second_word in self.bigrams[word].keys()],
            p=[frequency for frequency in self.bigrams[word].values()])

    def generate_n_words(self, base_word: str, n_words: int) -> list:
        """Given a word, generates n number of words based on supplied bigrams"""
        generated_words = []
        for _ in range(n_words):
            next_word = self._generate_next_word(base_word)
            generated_words.append(next_word)
            base_word = next_word
        return generated_words

if __name__ == "__main__":
    main()