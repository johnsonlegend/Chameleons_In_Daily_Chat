"""
Final Project for EECS 595
Implement word embedding method, part-of-speech pattern method on calculating sentence similarity.

@ chameleon_johson.py
@ johson
@ Nov 29, 2018
"""

import sys
import numpy as np
import pickle
from glob import glob
import nltk
import gensim.downloader as api
import matplotlib.pyplot as plt


DEBUG = 1
TESTFILE = 61
TESTLINE = 3000
COSINE_SIMILAR_THRESHOLD = 0.8


def read_clean_data(fname):
	"Read in clean data. Return object data."
	with open(fname, 'rb') as f:
		obj = pickle.load(f)

	return obj


def split_conversation(obj):
	"Split conversation by each speaker. Return splited conversation data."
	# Store first line in conversation
	prev_spk = obj[0][0]
	conv = [obj[0]]

	# Start at the second line, check if the speaker does not change
	for line in obj[1:]:
		curr_spk = line[0]
		if curr_spk == prev_spk:
			conv[-1][-1] += line[-1]
		else:
			conv.append(line)
			prev_spk = curr_spk

	return conv


def calculate_word2vec_similar(conv, model, stopwords):
	"""
	Calculate sentence similarity based on word embedding.
	Return top 3 word similarity scores on each pair of sentence.
	"""
	n = len(conv)
	similar_score_top3_list = []

	for i in range(n - 1):
		senten1 = conv[i][1]
		senten2 = conv[i+1][1]
		word_1 = []
		word_2 = []
		senten_vec1 = []
		senten_vec2 = []

		# Calculate word2vec
		for word in senten1:
			if word not in stopwords:
				try:
					senten_vec1.append(model[word.lower()])
					word_1.append(word.lower())
				except Exception as e:
					e = 0
		for word in senten2:
			if word not in stopwords:
				try:
					senten_vec2.append(model[word.lower()])
					word_2.append(word.lower())
				except Exception as e:
					e = 0

		# Store word pair, in order to check whether the top 3 similarity scores make sense
		word_pair = []
		for word1 in word_1:
			for word2 in word_2:
				word_pair.append((word1, word2))

		# Calculate similarity score
		similar_score = []
		for vec1 in senten_vec1:
			for vec2 in senten_vec2:
				# Cosine similarity
				cosine_similar = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
				# if cosine_similar == 1.0:
				# 	cosine_similar = 0
				similar_score.append(cosine_similar)

		# Choose top 3 scores
		similar_score_top3 = sorted(similar_score, reverse=True)[:3]
		if (len(similar_score_top3) == 3):
			# if (similar_score_top3[-1] >= COSINE_SIMILAR_THRESHOLD):
			similar_score_top3_list.append(similar_score_top3)

		for score in similar_score_top3:
			idx = similar_score.index(score)
			# print(word_pair[idx])

	return similar_score_top3_list


def calculate_senten_len_similar(conv):
	"Calculate sentence similarity based on speech pattern (sentence length)."
	n = len(conv)
	senten_len_score = []

	for i in range(n - 1):
		senten1 = conv[i][1]
		senten2 = conv[i+1][1]
		len1 = len(senten1)
		len2 = len(senten2)

		if len1 == 0 or len2 == 0:
			continue

		senten_len_score.append(1.0 - (abs(len1 - len2) / (len1 + len2)))

	if (len(senten_len_score) == 0):
		return 0

	return sum(senten_len_score) / len(senten_len_score)


def calculate_pos_similar(conv):
	"Calculate sentence similarity based on speech pattern (part-of-speech)."
	n = len(conv)
	pos_score = []

	for i in range(n - 1):
		senten1 = conv[i][1]
		senten2 = conv[i+1][1]

		tag1 = nltk.pos_tag(senten1)
		tag2 = nltk.pos_tag(senten2)

		# Store in dictionary
		tag_count1 = {}
		tag_count2 = {}
		for tag in tag1:
			if tag[1][:2] in tag_count1.keys():
				tag_count1[tag[1][:2]] += 1
			else:
				tag_count1[tag[1][:2]] = 1
		for tag in tag2:
			if tag[1][:2] in tag_count2.keys():
				tag_count2[tag[1][:2]] += 1
			else:
				tag_count2[tag[1][:2]] = 1

		if len(tag_count1) == 0 or len(tag_count2) == 0:
			continue

		# Calculate similar score
		count = 0
		for key in tag_count1.keys():
			if key in tag_count2.keys():
				count += tag_count1[key] * tag_count2[key]

		count /= np.sqrt(np.sum(np.square(list(tag_count1.values()))))
		count /= np.sqrt(np.sum(np.square(list(tag_count2.values()))))
		pos_score.append(count)

	# Choose top 20 scores
	pos_score_top20 = sorted(pos_score, reverse=True)[:20]

	return pos_score_top20


def main():
	"Main function."
	clean_data_files = glob('TRN_output/*.pkl')
	if DEBUG:
		clean_data_files = clean_data_files[:TESTFILE]

	# Load pretrained gensim model
	model = api.load("glove-twitter-25")

	# print(model["cold"])
	# print(model["big"])
	# print(np.dot(model["cold"], model["big"]) / (np.linalg.norm(model["cold"]) * np.linalg.norm(model["big"])))

	# Read labels
	with open('label', 'r') as f:
		lines = f.readlines()
	labels = [line.rstrip('\n').split()[2] for line in lines]
	yes_labels = [i for i in range(len(labels)) if labels[i] == 'yes']
	no_labels = [i for i in range(len(labels)) if labels[i] == 'no']
	no_labels.remove(24)

	# Read stop words
	with open('stopwords', 'r') as f:
		lines = f.readlines()
	stopwords = [line.rstrip('\n') for line in lines]

	WORD_SIMILAR_SCORE = []
	SENTEN_SIMILAR_SCORE = []
	POS_SIMILAR_SCORE = []

	# Read clean data
	for fname in clean_data_files:
		obj = read_clean_data(fname)

		if len(obj) == 0:
			continue

		if DEBUG:
			obj = obj[:TESTLINE]

		# Get conversation
		conv = split_conversation(obj)

		# Method 1: Calculate word2vec similarity
		similar_score_top3_list = calculate_word2vec_similar(conv, model, stopwords)
		if (len(similar_score_top3_list) == 0):
			WORD_SIMILAR_SCORE.append(0)
		else:
			WORD_SIMILAR_SCORE.append(np.sum(np.sum(similar_score_top3_list)) / len(similar_score_top3_list))
		# print(np.sum(np.sum(similar_score_top3_list)) / len(similar_score_top3_list))

		# Method 2: Speech pattern (sentence length, part-of-speech types)
		senten_len_similar = calculate_senten_len_similar(conv)
		SENTEN_SIMILAR_SCORE.append(senten_len_similar)
		# print(senten_len_similar)

		pos_similar_top20 = calculate_pos_similar(conv)
		if (len(pos_similar_top20) == 0):
			POS_SIMILAR_SCORE.append(0)
		else:
			POS_SIMILAR_SCORE.append(sum(pos_similar_top20) / len(pos_similar_top20))
		# print(sum(pos_similar_top20) / len(pos_similar_top20))

	# Check result by labels and plot
	WORD_SIMILAR_SCORE_YES = [WORD_SIMILAR_SCORE[i] for i in yes_labels]
	WORD_SIMILAR_SCORE_NO = [WORD_SIMILAR_SCORE[i] for i in no_labels]
	plt.plot(yes_labels, WORD_SIMILAR_SCORE_YES, 'rx-', label='Close')
	plt.plot(no_labels, WORD_SIMILAR_SCORE_NO, 'bx-', label='Non-close')
	# plt.ylim(0,3)
	plt.xlabel('Conversation')
	plt.ylabel('Word embedding similarity score')
	plt.legend()
	plt.savefig('word_similar_score.png')
	plt.clf()
	print("close relationship: " + str(sum(WORD_SIMILAR_SCORE_YES) / len(WORD_SIMILAR_SCORE_YES)))
	print("non-close relationship: " + str(sum(WORD_SIMILAR_SCORE_NO) / len(WORD_SIMILAR_SCORE_NO)))

	SENTEN_SIMILAR_SCORE_YES = [SENTEN_SIMILAR_SCORE[i] for i in yes_labels]
	SENTEN_SIMILAR_SCORE_NO = [SENTEN_SIMILAR_SCORE[i] for i in no_labels]
	plt.plot(yes_labels, SENTEN_SIMILAR_SCORE_YES, 'rx-', label='Close')
	plt.plot(no_labels, SENTEN_SIMILAR_SCORE_NO, 'bx-', label='Non-close')
	plt.ylim(0,1)
	plt.xlabel('Conversation')
	plt.ylabel('Sentence length similarity score')
	plt.legend()
	plt.savefig('sentence_similar_score.png')
	plt.clf()
	print("close relationship: " + str(sum(SENTEN_SIMILAR_SCORE_YES) / len(SENTEN_SIMILAR_SCORE_YES)))
	print("non-close relationship: " + str(sum(SENTEN_SIMILAR_SCORE_NO) / len(SENTEN_SIMILAR_SCORE_NO)))

	POS_SIMILAR_SCORE_YES = [POS_SIMILAR_SCORE[i] for i in yes_labels]
	POS_SIMILAR_SCORE_NO = [POS_SIMILAR_SCORE[i] for i in no_labels if POS_SIMILAR_SCORE[i] > 0]
	no_labels.remove(29)
	plt.plot(yes_labels, POS_SIMILAR_SCORE_YES, 'rx-', label='Close')
	plt.plot(no_labels, POS_SIMILAR_SCORE_NO, 'bx-', label='Non-close')
	plt.ylim(0,1)
	plt.xlabel('Conversation')
	plt.ylabel('Part-of-speech similarity score')
	plt.legend()
	plt.savefig('pos_similar_score.png')
	plt.clf()
	print("close relationship: " + str(sum(POS_SIMILAR_SCORE_YES) / len(POS_SIMILAR_SCORE_YES)))
	print("non-close relationship: " + str(sum(POS_SIMILAR_SCORE_NO) / len(POS_SIMILAR_SCORE_NO)))


if __name__ == '__main__':
	main()

