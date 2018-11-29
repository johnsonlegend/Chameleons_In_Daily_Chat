import pickle
import sys
import re

#token: list of tags
liwc = {}

def load_liwc():
	f = open("LIWC.2015.all","r")
	line = f.readline()
	while line:
		token = re.sub(r'\W+', '', line.split(" ,")[0])
		cate = re.sub(r'\W+', '', line.split(" ,")[1]) 
		if liwc.get(token) == None:
			liwc[token] = []
		liwc[token].append(cate)

		line = f.readline()
	f.close()

def load_conversation():
	f = open("TRN_output/SBC001.pkl","rb")
	conv = pickle.load(f)
	f.close()
	return conv

def combine_conv(conv):
	conv_combined = []
	line = conv[0]
	for idx in range(len(conv) - 1):
		if conv[idx][0] == conv[idx + 1][0]:
			line[1] += conv[idx + 1][1]
		else:
			conv_combined.append(line)
			line = conv[idx + 1]
	conv_combined.append(line)
	return conv_combined

if __name__ == '__main__':
	load_liwc()
	conv = load_conversation()
	conv_combined = combine_conv(conv)
	


	