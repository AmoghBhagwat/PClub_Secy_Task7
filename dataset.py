import torch
import pandas as pd
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.words = self.load_words()
        self.unique_words = self.get_unique_words()

        self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv('train1.csv')
        train_df = train_df[train_df['clickbait'] == 1]
        
        train_df['headline'] = train_df['headline'].str.replace(r'[^A-Za-z0-9 ]+', '')
        train_df['headline'] = train_df['headline'].str.lower()
        text = train_df['headline'].str.cat(sep=' ')
        
        train_2 = pd.read_csv('train2.csv')
        train_2 = train_2[train_2['label'] == 'clickbait']
        text2 = train_2['title'].str.replace(r'[^A-Za-z0-9 ]+', '')
        text2 = text2.str.lower()
        text2 = text2.str.cat(sep=' ')

        final = text2[0:8000].split(' ')# + text[0:2500].split(' ');
        print(final)
        return final
        
    def get_unique_words(self):
        return np.unique(self.words)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
        )
