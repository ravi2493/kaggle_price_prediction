import csv
import numpy as np
from nltk import word_tokenize
import pandas as pd

def split_words(sentence):
    return sentence.split()

def update(word2Id,Id2Word,words):
    keys = list(word2Id.keys())
    for word in words:
        if word not in keys:
            word2Id[word] = len(keys)
            Id2Word[len(keys)] = word
            keys.append(word)

def batch_iter(data, batch_size, epochs, Isshuffle=True):
    num_batches = int((len(data)-1)/batch_size)
    data_size = len(data)
    print("size of data"+str(data_size)+"---"+str(len(data)))
    for ep in range(epochs):
        if Isshuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            yield shuffled_data[start_index:end_index]


def text2Ids(words,word2Id,max_sequence_len):
    a = np.zeros(max_sequence_len,dtype=int)
    keys = word2Id.keys()
    for i,word in enumerate(words):
        if word in keys:
            a[i] = word2Id[word]
        else:
            a[i] = word2Id['--unknown--']
    return a

def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)

def load_train_data(path):
    data = pd.read_table(path)
    data = handle_missing(data)
    word2Id = {}
    Id2Word = {}
    dt = {}
    dt['name'] = []
    dt['item_condition_id'] = []
    dt['category_name'] = []
    dt['brand_name'] = []
    dt['price'] = []
    dt['shipping'] = []
    dt['item_description'] = []
    max_name_len = -1
    max_cat_name_len = -1
    max_item_desc_len = -1
    word2Id['empty'] = 0
    for i,row in enumerate(data.values):
        if i == 0:
            continue
        splt = row[1:]
        if i%10000 == 0:
            print(str(i)+"__0")
            # break
        name_words = split_words(splt[0])
        if max_name_len < len(name_words):
            max_name_len = len(name_words)
        update(word2Id, Id2Word, name_words)
        dt['name'].append(name_words)

        dt['item_condition_id'].append(splt[1])

        cat_name_words = split_words(splt[2])
        if max_cat_name_len < len(cat_name_words):
            max_cat_name_len = len(cat_name_words)
        update(word2Id, Id2Word, cat_name_words)
        dt['category_name'].append(cat_name_words)

        dt['brand_name'].append("".join(split_words(splt[3])))
        update(word2Id, Id2Word, ["".join(split_words(splt[3]))])

        dt['price'].append(splt[4])
        dt['shipping'].append(splt[5])

        item_desc_words = split_words(splt[6])
        if max_item_desc_len < len(item_desc_words):
            max_item_desc_len = len(item_desc_words)
        update(word2Id, Id2Word, item_desc_words)
        dt['item_description'].append(item_desc_words)

    data_len = len(dt['name'])
    name_arr = np.zeros([data_len,max_name_len])
    name_arr_len = np.zeros([data_len])
    item_cont_arr = np.zeros([data_len,1])
    cat_name_arr = np.zeros([data_len,max_cat_name_len])
    cat_name_arr_len = np.zeros([data_len])
    brand_name_arr = np.zeros([data_len,1])
    price_arr = np.zeros([data_len,1])
    shipping_arr = np.zeros([data_len,1])
    item_desc_arr = np.zeros([data_len,max_item_desc_len])
    item_desc_arr_len = np.zeros([data_len])
    for i in range(data_len):
        if i%10000 == 0:
            print(str(i)+"__1")
        name_arr[i,:] = text2Ids(dt['name'][i],word2Id,max_name_len)
        name_arr_len[i] = len(dt['name'][i])
        item_cont_arr[i,0] = int(dt['item_condition_id'][i])
        cat_name_arr[i,:] = text2Ids(dt['category_name'][i],word2Id,max_cat_name_len)
        cat_name_arr_len[i] = len(dt['category_name'][i])
        if len(dt['brand_name'][i]) > 0:
            brand_name_arr[i,0] = word2Id[dt['brand_name'][i]]
        else:
            brand_name_arr[i,0] = word2Id['empty']
        price_arr[i,0] = float(dt['price'][i])
        shipping_arr[i,0] = int(dt['shipping'][i])
        item_desc_arr[i,:] = text2Ids(dt['item_description'][i], word2Id , max_item_desc_len)
        item_desc_arr_len[i] = len(dt['item_description'][i])
    mp = {}
    mp['max_name_len'] = max_name_len
    mp['max_cat_name_len'] = max_cat_name_len
    mp['max_item_desc_len'] = max_item_desc_len
    mp['vocab_size'] = len(word2Id.keys())
    return list(zip(name_arr,name_arr_len,item_cont_arr,cat_name_arr,cat_name_arr_len,
                    brand_name_arr,price_arr,shipping_arr,item_desc_arr,item_desc_arr_len)),mp

def chunks(path):
    chunksize = 100000
    for chunk in pd.read_csv(path, chunksize=chunksize, error_bad_lines=False):
        d1 = load_train_data(chunk.values)

# chunks(path=r'C:\Users\pravi\Downloads\train.tsv')
# load_train_data(path=r'C:\Users\pravi\Downloads\train.tsv')