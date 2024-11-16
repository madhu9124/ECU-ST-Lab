import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag
from textstat import textstat

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to read sentences from files
def read_sentences(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Function to ensure each sentence ends with a question mark
def ensure_question_mark(sentences):
    return [sentence if sentence.endswith('?') else sentence + '?' for sentence in sentences]

# Function to compute cosine similarity
def compute_cosine_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Function to compute semantic similarity
def compute_semantic_similarity(sentence1, sentence2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([sentence1, sentence2])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

# Function to analyze tone
def analyze_tone(sentence):
    analysis = TextBlob(sentence)
    return analysis.sentiment.polarity

# Function to analyze sentiment
def analyze_sentiment(sentence):
    analysis = TextBlob(sentence)
    return analysis.sentiment.polarity

# Function to compute linguistic and lexical diversity
def compute_diversity(sentence):
    tokens = word_tokenize(sentence)
    if len(tokens) == 0:
        return 0, 0  # Avoid division by zero
    lexical_diversity = len(set(tokens)) / len(tokens)
    readability_score = textstat.flesch_reading_ease(sentence)
    return lexical_diversity, readability_score

# Function for Named Entity Recognition
def perform_ner(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    entities = []
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))
    return entities

# Function to calculate total unique entities
def calculate_total_unique_entities(ner1, ner2):
    set1 = set(ner1)
    set2 = set(ner2)
    unique_to_1 = set1 - set2
    unique_to_2 = set2 - set1
    return len(unique_to_1) + len(unique_to_2)

# Read sentences from files- soure test case
sentences_file1 = read_sentences('/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/sourcetestcase_without_sensitive_attribute.txt')
# Read sentences from files- follow-up test case
sentences_file2 = read_sentences('/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/follow-up test case_Add_single_Attribute.txt')

# Check each sentence ends with a question mark
sentences_file1 = ensure_question_mark(sentences_file1)
sentences_file2 = ensure_question_mark(sentences_file2)

# Check if the number of sentences match
if len(sentences_file1) != len(sentences_file2):
    raise ValueError("The number of sentences in file1.txt and file2.txt do not match.")

# Create directories to store the results
os.makedirs('results/cosine_similarity', exist_ok=True)
os.makedirs('results/semantic_similarity', exist_ok=True)
os.makedirs('results/tone_analysis', exist_ok=True)
os.makedirs('results/sentiment_analysis', exist_ok=True)
os.makedirs('results/lexical_diversity', exist_ok=True)
os.makedirs('results/linguistic_diversity', exist_ok=True)
os.makedirs('results/ner_diversity', exist_ok=True)

# Initialize result strings
cosine_similarity_results = []
semantic_similarity_results = []
tone_analysis_results = []
sentiment_analysis_results = []
lexical_diversity_results = []
linguistic_diversity_results = []
ner_diversity_results = []

# Looping through the sentences and compute metrics
for i, (sentence1, sentence2) in enumerate(zip(sentences_file1, sentences_file2), start=1):
    if not sentence1 or not sentence2:
        continue 
    
    cosine_sim = compute_cosine_similarity(sentence1, sentence2)
    semantic_sim = compute_semantic_similarity(sentence1, sentence2)
    tone1 = analyze_tone(sentence1)
    tone2 = analyze_tone(sentence2)
    tone_diff = abs(tone1 - tone2)
    sentiment1 = analyze_sentiment(sentence1)
    sentiment2 = analyze_sentiment(sentence2)
    sentiment_diff = abs(sentiment1 - sentiment2)
    diversity1, readability1 = compute_diversity(sentence1)
    diversity2, readability2 = compute_diversity(sentence2)
    lexical_diversity_diff = abs(diversity1 - diversity2)
    readability_diff = abs(readability1 - readability2)
    
    # NER calculation
    ner1 = perform_ner(sentence1)
    ner2 = perform_ner(sentence2)
    total_unique_entities = calculate_total_unique_entities(ner1, ner2)

    # Append the results to the lists
    cosine_similarity_results.append(f'Sentence {i}: {cosine_sim}\n')
    semantic_similarity_results.append(f'Sentence {i}: {semantic_sim}\n')
    tone_analysis_results.append(f'Sentence {i}: {tone_diff}\n')
    sentiment_analysis_results.append(f'Sentence {i}: {sentiment_diff}\n')
    lexical_diversity_results.append(f'Sentence {i}: {lexical_diversity_diff}\n')
    linguistic_diversity_results.append(f'Sentence {i}: {readability_diff}\n')
    ner_diversity_results.append(f'Sentence {i}: {total_unique_entities}\n')

# Write the results to text files
with open('results/cosine_similarity/cosine_similarity.txt', 'w') as file:
    file.writelines(cosine_similarity_results)

with open('results/semantic_similarity/semantic_similarity.txt', 'w') as file:
    file.writelines(semantic_similarity_results)

with open('results/tone_analysis/tone_analysis.txt', 'w') as file:
    file.writelines(tone_analysis_results)

with open('results/sentiment_analysis/sentiment_analysis.txt', 'w') as file:
    file.writelines(sentiment_analysis_results)

with open('results/lexical_diversity/lexical_diversity.txt', 'w') as file:
    file.writelines(lexical_diversity_results)

with open('results/linguistic_diversity/linguistic_diversity.txt', 'w') as file:
    file.writelines(linguistic_diversity_results)

with open('results/ner_diversity/ner_total_unique_entities.txt', 'w') as file:
    file.writelines(ner_diversity_results)

print(f"Processed {len(sentences_file1)} sentence pairs successfully.")
