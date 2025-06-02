# chatbot_logic.py
# ----------------
# A standalone module for a Level-1 TF-IDF rule-based chatbot.
# Contains all necessary functions and classes to load a cleaned Q-A dataset,
# build a TF-IDF vectorizer, and return answers to user queries.
#
# You can import this module into Flask, Streamlit, FastAPI, or any other
# framework. Example:
#
#   from chatbot_logic import Chatbot
#   bot = Chatbot(csv_path="path/to/clean_conversation_dataset.csv")
#   response = bot.get_answer("hello, how are you?")
#
# To run interactively as a script:
#   python chatbot_logic.py
#
# Make sure you have installed:
#   pandas, scikit-learn

import pandas as pd
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text: str) -> str:
    """
    Normalize a string so that it matches the preprocessing applied to the
    Q-A dataset. Steps:
      1. Strip leading/trailing whitespace.
      2. Convert to lowercase.
      3. Apply Unicode NFKD normalization (decompose accented characters).
      4. Encode to ASCII, ignoring non-ASCII characters.
      5. Collapse multiple whitespace characters into a single space.
    """
    # 1) Strip + lowercase
    s = text.strip().lower()
    # 2) Unicode NFKD normalization
    s = unicodedata.normalize("NFKD", s)
    # 3) Encode to ASCII and ignore any non-ASCII bytes
    s = s.encode("ascii", "ignore").decode("ascii")
    # 4) Collapse multiple whitespace (spaces, tabs, newlines) into a single space
    s = re.sub(r"\s+", " ", s).strip()
    return s


class Chatbot:
    """
    A TF-IDF based rule-driven chatbot.

    Attributes:
        df (pd.DataFrame): DataFrame containing 'question' and 'answer' columns.
        questions (List[str]): List of all cleaned questions.
        answers (List[str]): List of all corresponding answers.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): TF-IDF matrix of all questions.
    """

    def __init__(self,
                 csv_path: str = "D:/PROJECTS/Version Control/CHATBOT/building-smart-bots-from-scratch/data/clean_conversation_dataset.csv",
                 text_column: str = "question",
                 answer_column: str = "answer",
                 stop_words: str = "english"):
        """
        Initialize the chatbot by loading a cleaned CSV, vectorizing questions,
        and building a TF-IDF matrix.

        Args:
            csv_path (str): Path to a CSV with at least the specified text_column and answer_column.
            text_column (str): Column name containing questions.
            answer_column (str): Column name containing answers.
            stop_words (str or list): Passed to TfidfVectorizer (e.g. "english" or list of tokens).
        """
        # 1) Load cleaned dataset
        self.df = pd.read_csv(csv_path)

        # 2) Extract questions and answers
        #    We assume questions/answers are already normalized (lowercase, ASCII-only).
        self.questions = self.df[text_column].astype(str).tolist()
        self.answers = self.df[answer_column].astype(str).tolist()

        # 3) Instantiate TF-IDF vectorizer (removing stop-words)
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

        # 4) Fit the vectorizer on all questions and transform them into TF-IDF vectors
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def get_answer(self, user_input: str) -> str:
        """
        Given a raw user input string, normalize and vectorize it,
        compute cosine similarity against all stored question vectors,
        and return the answer corresponding to the best-matching question.

        Args:
            user_input (str): Raw text from the user.

        Returns:
            str: The answer with the highest cosine similarity to user_input.
        """
        # 1) Normalize the raw input
        clean_input = normalize_text(user_input)

        # 2) Transform the normalized input into a TF-IDF vector
        user_vec = self.vectorizer.transform([clean_input])

        # 3) Compute cosine similarities (1 × n_questions)
        sims = cosine_similarity(user_vec, self.tfidf_matrix)

        # 4) Identify index of most similar question
        best_idx = sims.argmax()

        # 5) Return the corresponding answer
        return self.answers[best_idx]


if __name__ == "__main__":
    """
    If this script is run directly, it will launch a simple command-line chat loop.
    Type 'exit' to quit.
    """
    # Create a Chatbot instance (point csv_path to your cleaned CSV)
    bot = Chatbot(csv_path="D:/PROJECTS/Version Control/CHATBOT/building-smart-bots-from-scratch/data/clean_conversation_dataset.csv")

    print("🤖 Rule_Based Chatbot is ready! (Type 'exit' to quit)\n")

    while True:
        user_text = input("You: ")
        if user_text.strip().lower() == "exit":
            print("Bot: Goodbye! 👋")
            break
        if not user_text.strip():
            print("Bot: Please type something or 'exit' to quit.")
            continue

        response = bot.get_answer(user_text)
        print("Bot:", response)
