import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# before you run this file, rename in the file in results/cosine_similarity to sentence_score.txt, do the same for all the metrics.
# Define the paths to the folders
folders = ['/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/results/cosine_similarity', '/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/results/lexical_diversity', '/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/results/semantic_similarity', '/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/results/linguistic_diversity', '/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/results/ner_diversity', '/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/results/tone_analysis','/home/madhy/Documents/fairness_testing/Prioritize MRs/MRs/addition/results/sentiment_analysis']
file_name = 'sentence_score.txt'
#file_names = ['sentence_score.txt']  # Assuming the files are named the same in each folder

# Initialize lists to hold the scores from each folder
all_scores = {folder: [] for folder in folders}

# Read the scores from the text files
for folder in folders:
    with open(os.path.join(folder, file_name), 'r') as file:
        print(folder)
        scores = [float(line.strip().split(': ')[1]) for line in file.readlines()]
        all_scores[folder] = scores

# Transpose the scores to get a list of scores for each sentence
scores_per_sentence = list(zip(*[all_scores[folder] for folder in folders]))

# Normalize the scores
scaler = MinMaxScaler()
normalized_scores = scaler.fit_transform(scores_per_sentence)

# Define the equal weights for each attribute
weights = np.array([1/len(folders)] * len(folders))  # 1/5 for each of the 5 folders

# Calculate weighted differences for each sentence
# np.dot(weights, sentence_scores) computes the weighted sum of the normalized scores for each sentence
weighted_differences = [np.dot(weights, sentence_scores) for sentence_scores in normalized_scores]

# Sum the weighted differences to get the total diversity score
total_diversity_score = np.sum(weighted_differences)

#print("Normalized Scores for Each Sentence:", normalized_scores)
#print("Weighted Differences for Each Sentence:", weighted_differences)
print("Total Diversity Score:", total_diversity_score)
