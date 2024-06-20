# Download the dataset
with open('shakespeare_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of our condensed shakespeare:", len (text))

print(text[:1000])

# Figure out all the characters the model can see or emit
model_vocabulary = sorted(list(set(text)))
vocab_size = len(model_vocabulary)
print(''.join(model_vocabulary))
print(vocab_size)

# We need to convert the full text into a sequence of integers (which are all elements of our vocabulary)
# We need to make an encoder and a decoder to do this.
# first make a mapping both ways
char_to_int = { character:i for i,character in enumerate(model_vocabulary) }
int_to_char = { i:character for i,character in enumerate(model_vocabulary) }

# second, create functions to encode and decode
# Suppose we get a string, take each char in that string and return its corresponding int
encode = lambda s: [char_to_int[c] for c in s]

# Suppowe we get a list of ints, take each element in that list and give it to the int to char dict. 
# Produce a list of characters, then combine them into one string
decode = lambda l: ''.join([int_to_char[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))

# encode all of shakespeare
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# Split off training and validation data
cutoff = int(0.9*len(data))
train_data = data[:cutoff]
val_data = data[cutoff:]

#confirm that we've split the data 90:10
print(len(train_data))
print(len(val_data))
 
# set the context size of the transformer
context_size = 8

print(train_data[:context_size+1])

x = train_data[:context_size]
y = train_data[1:context_size+1]
for i in range(context_size):
    context = x[:i+1]
    target = y[i]
    print(f"We want to use {context} to predict {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences of integers do we process in parallel?
context_size = 8 # What is the max amount of integers we use to predict the next integer?

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # generate a tensor. populate it with random integers from 0 to just under the dataset length. Put them in a (4,1) matrix.
    start_index_predictor = torch.randint(len(data) - context_size, (batch_size,))
    predictors = torch.stack([data[i:i+context_size] for i in start_index_predictor])
    targets = torch.stack([data[i+1:i+context_size+1] for i in start_index_predictor])
    return predictors, targets

predictor_b, target_b = get_batch('train')
print('inputs:')
print(predictor_b.shape)
print(predictor_b)
print('outputs:')
print(target_b.shape)
print(target_b)

import torch.nn as nn

class BigramLanguageModel(nn.Module):
    def __ini__(self, vocab_size):
        super().__init__()
        #Make a lookup table for each letter. Each letter gets a vector with the vocab 
        # length because each value is essentially the probability of the next letter, 
        # for all possible letters. Eg. column 1 might represent 'a'. (1,1) is the odds 
        # of 'a' coming after 'a' and (1,26) the odds of 'z' after 'a'
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)