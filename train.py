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
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #Make a lookup table for each letter. Each letter gets a vector with the vocab 
        # length because each value is essentially the probability of the next letter, 
        # for all possible letters. Eg. column 1 might represent 'a'. (1,1) is the odds 
        # of 'a' coming after 'a' and (1,26) the odds of 'z' after 'a'
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_tensor, targets=None):
        # when we pass our input tensor to the embedding table, each integer 
        # (think character) is then matched to its embedding vector. Hence we obtain a 
        # new dimension to our previously only batch x context_len matrix. Now we get 
        # batch x context_len x vocab length. Each integer now becomes a probability 
        # distribution over the next word.
        logits = self.token_embedding_table(input_tensor) #(batch x context_size x vocab_length)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape #This outputs integer values for each of the dimensions of our logits tensor

            logits = logits.view(B*T, C) # make a long single file line of probability distributions to predict the targets

            targets = targets.view(B*T) #

            loss = torch.nn.functional.cross_entropy(logits, targets)
        
        return logits, loss
        # the logits that we return will be a single file list of probability distributions trying to predict the next character
        # The loss that we return will be the average loss of these probability distributions in actually predicting the target next character
    
    def generate(self, input_tensor, max_new_tokens):
        # input_tensor is the (batch x context_length) matrix
        for _ in range(max_new_tokens): # '_' just means that we need to run the loop as many times as specified, but we can ignore the index when doing so
            logits, loss = self(input_tensor) #in pytorch when we call the __call__ method (which is what happens when we use an object as a function), we get forward(). Basically this step calls forward on our input tensor
            #Just look at the prob dist. we have for the very last token in the context window
            final_token_logit = logits[:, -1, :] #batch x vocab_length
            #Take the softmax of these logits to get the probabilities of the next token being each of the possible values in the vocab
            probs = torch.nn.functional.softmax(final_token_logit, dim = 1) #batch x vocab_length
            #Sample from this prob dist; get a realization of the next token
            next_token = torch.multinomial(probs, num_samples=1) # batch x 1
            #Append this new token to our input tensor
            output_tensor = torch.cat((input_tensor, next_token), dim = 1) #batch x context_length+1
            input_tensor = output_tensor
        return output_tensor


    
model = BigramLanguageModel(vocab_size)
logits, loss = model(predictor_b, target_b)
print(logits.shape)
print(loss)

# This below generates a new sequence of tokens. We start with a single batch. The context window is just a single token: the integer zero. 
# Because "generate" creates a batch x context_length+max_tokens matrix, we need to select just the first (and only) batch dimension. We then convert this vector to a python list.
# model.generate(input_tensor=torch.zeros((1,1), dtype = torch.long), max_new_tokens=100)[0].tolist()

print(decode(model.generate(input_tensor=torch.zeros((1,1), dtype = torch.long), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

#increase the batch size -- now we want to actually optimize things quickly, not just create intuition
batch_size = 32
for steps in range(10000):

    #sample batch of data
    input_batch, target_batch = get_batch('train')

    logits, loss = model(input_batch, target_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#print our loss at the end of training
print(loss.item())

#Generate 100 new tokens
print(decode(model.generate(input_tensor=torch.zeros((1,1), dtype = torch.long), max_new_tokens=100)[0].tolist()))
