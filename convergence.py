import pickle
import sys
import re
import numpy as np

#liwc: [token : list of tags]
liwc = {}
fw_type = {
	'PRONOUN' : 0,
	'IPRON' : 1,
	'ARTICLE' : 2,
	'PREP' : 3,
	'AUXVERB' : 4,
	'ADVERB' : 5,
	'CONJ' : 6,
	'NEGATE' : 7,
	'QUANT' : 8
}
fw_len = len(fw_type)
label = []

def load_label():
	f = open("label","r")
	line = f.readline()
	while line:
		if line.split(" ")[2] == "yes\n":
			label.append(True)
		else:
			label.append(False)
		line = f.readline()
	f.close()

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

def load_conversation(num):
	if num < 10:
		file = "TRN_output/SBC00" + str(num) + ".pkl"
	else:
		file = "TRN_output/SBC0" + str(num) + ".pkl"
	f = open(file,"rb")
	conv = pickle.load(f)
	f.close()
	return conv

# combine adjacent lines by the same person
def combine_conv(conv):
	conv_combined = []
	line = conv[0]
	for idx in range(len(conv) - 1):
		if conv[idx][0] == conv[idx + 1][0]:
			line[1] += conv[idx + 1][1]
		else:
			if len(line[1]) > 5:
				conv_combined.append(line)
			line = conv[idx + 1]
	conv_combined.append(line)
	return conv_combined

# extract function words
def extract_fw(line):
	

	fw = np.zeros(fw_len)
	for w in line[1]:
		if liwc.get(w):
			for type in liwc[w]:
				if type in fw_type:
					fw[fw_type[type]] = 1

	return line[0], fw

def analysis(conv):
	conv_combined = combine_conv(conv)

	fw_avg = np.zeros(fw_len)
	fw_all = []
	for line in conv_combined:
		person, fw = extract_fw(line)
		fw_all.append(fw)
		fw_avg += fw

	fw_given = np.zeros(fw_len)

	for i in range(len(fw_all) - 1):
		for tag_idx in range(fw_len):
			if fw_all[i][tag_idx] == 1:
				fw_given[tag_idx] += fw_all[i + 1][tag_idx]
	fw_avg = sum(fw_all)/len(fw_all)

	fw_given_avg = fw_given / (sum(fw_all) - fw_all[-1])
	# print(fw_given_avg - fw_avg)

	return fw_given_avg - fw_avg

if __name__ == '__main__':
	load_label()
	load_liwc()
	num_close = sum(label)

	# close relationship
	res = np.zeros(fw_len)
	for num in range (1,61):
		if label[num - 1] == True:
			conv = load_conversation(num)
			res += analysis(conv)
	print(res/num_close)

	# non-close relationship

	res = np.zeros(fw_len)
	for num in range (1,61):
		if label[num - 1] == False:
			conv = load_conversation(num)
			res += analysis(conv)
	print(res/(60 - num_close))
	
