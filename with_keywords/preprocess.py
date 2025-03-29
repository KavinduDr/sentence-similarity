import re
import string
import nltk
import Levenshtein
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize preprocessing tools
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Cleans and lemmatizes text."""
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(f"[{string.punctuation}]", " ", text)  # Remove punctuation
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized)

def calculate_features(correct, keywords, student):
    """Calculates similarity scores between correct answer and student answer."""
    processed_correct = preprocess_text(correct)
    processed_keywords = " ".join([preprocess_text(k) for k in keywords])
    processed_student = preprocess_text(student)
    reference = processed_correct + " " + processed_keywords

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([reference, processed_student])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    set_ref = set(reference.split())
    set_stu = set(processed_student.split())
    jaccard = len(set_ref.intersection(set_stu)) / len(set_ref.union(set_stu)) if set_ref.union(set_stu) else 0

    lev = Levenshtein.distance(reference, processed_student)

    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([reference.split()], processed_student.split(), smoothing_function=smooth)

    return [cosine_sim, jaccard, 0, lev, 0.5, bleu]  # WMD & WordNet scores are placeholders