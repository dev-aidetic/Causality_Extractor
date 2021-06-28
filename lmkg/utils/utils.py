import random
import string
from typing import List
import re
import logging


def generate_key(length):
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


def clean_text(text: str, stop_words: List):
    """Clean the text for sentiment classification

    Args:
        text (str): Text to clean
        stop_words (List): List of stop words in the sentence

    Returns:
        [str]: Cleaned text
    """
    if text:
        text = text.lower()
        text = re.sub("'", "", text)
        text = re.sub("[^\w\s]", " ", text)
        text = re.sub(" \d+", " ", text)
        text = re.sub(" \\\w ", "", text)
        text = re.sub(" +", " ", text)
        text = text.replace("\n", "")
        text = text.strip()
        text = " ".join([span for span in text.split() if (span not in stop_words)])
        return text
    else:
        return None


def intialize_logging(log_name):
    log_format = "%(asctime)s %(levelname)s %(filename)s %(lineno)s: %(message)s"
    logging.basicConfig(format=log_format)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    return logger
