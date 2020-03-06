from nltk.corpus import stopwords
import nltk
from utils import clean_str
import sys

if len(sys.argv) != 2:
	sys.exit("Use: python remove_words.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'ag_news', 'dblp', 'TREC', 'WebKB','WebKB2'] # Modify
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# print(stop_words)

doc_content_list = []
f = open('data/corpus/' + dataset + '.txt', 'rb')

for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))
f.close()

word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        # word not in stop_words and word_freq[word] >= 5
        if dataset == 'mr' or dataset == 'TREC': # Modify
            doc_words.append(word)
        elif word not in stop_words and word_freq[word] >= 5:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    #if doc_str == '':
        #doc_str = temp  # 其实我觉得这个还是有必要的, 不知道为什么注释了
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)

f = open('data/corpus/' + dataset + '.clean.txt', 'w')

f.write(clean_corpus_str)
f.close()

min_len = 10000
aver_len = 0
max_len = 0 

f = open('data/corpus/' + dataset + '.clean.txt', 'r')

lines = f.readlines()
for line in lines:
    line = line.strip()
    temp = line.split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
    if len(temp) > max_len:
        max_len = len(temp)
f.close()
aver_len = 1.0 * aver_len / len(lines)
print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : ' + str(aver_len))
