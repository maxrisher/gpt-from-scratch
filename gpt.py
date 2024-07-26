import torch
import torch.nn as nn

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_sample_size = 200
n_embed = 32

torch.manual_seed(1337)



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

#this decorator means that pytorch will not store loss values (because this is just for evaluation of the model)
@torch.no_grad()
def estimate_loss():
    summary = {}
    model.eval()
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_sample_size)
        for iteration in range(eval_sample_size):
            predictors, targets = get_batch(split)
            logits, loss = model(predictors, targets)
            losses[iteration] = loss.item()
        summary[split] = losses.mean()

    #reset the model to training mode
    model.train()
    return summary

predictor_b, target_b = get_batch('train')
print('inputs:')
print(predictor_b.shape)
print(predictor_b)
print('outputs:')
print(target_b.shape)
print(target_b)

import torch.nn as nn
torch.manual_seed(1337)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x) # B x T x head_size
        q = self.query(x) # BxTxhead_size
        v = self.value(x) # BxTxhead_size

        #find affinity between k and q
        wei = q @ k.transpose(-2, -1) * C**-0.5 # we scale the wei values by the square root of the embedding space size. This is so that the values in wei do not get so large that softmax just selects the highest one
        # B x T x T

        #mask future values
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #use [:T, :T] to cut down the tril matrix to the size of the context window in our input tensor
        # B x T x T

        #softmax
        wei = nn.functional.softmax(wei, dim = -1)
        # B x T x T

        #values by weights
        out = wei @ v
        # B x T x head_size

        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)]) # create a Head with a given head size for all of the heads we want to create

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim =-1) #forward just calls forward on all of the heads. We then take these B x T x head_size tensors and stack them in the head_size direction
        # outputs BxTxhead_size*n_heads

class FeedForward(nn.Module):
    """ Simple linear layer followed by non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed), #here we add a linear layer to 'think' about the output of the previous layer
            nn.ReLU(), #add an activation layer here, to add non-linearities
        )
        
        # add an input layer, a hidden layer, and an output layer
    
    def forward(self, x):
        return self.net(x) 

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Make a lookup table for each letter. Each letter gets a vector with the vocab 
        # length because each value is essentially the probability of the next letter, 
        # for all possible letters. Eg. column 1 might represent 'a'. (1,1) is the odds 
        # of 'a' coming after 'a' and (1,26) the odds of 'z' after 'a'
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadedAttention(4, n_embed//4) #split whatever our n_embed is across 4 different heads of attention. Round down when the division is not even.
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) #Here we scale down the logits to be embedded in n_embed dimensions. Then we scale them back up to reach the number of dimensions in the full vocab size. 

    def forward(self, input_tensor, targets=None):
        B, T = input_tensor.shape

        # when we pass our input tensor to the embedding table, each integer 
        # (think character) is then matched to its embedding vector. Hence we obtain a 
        # new dimension to our previously only batch x context_len matrix. Now we get 
        # batch x context_len x vocab length. Each integer now becomes a probability 
        # distribution over the next word.
        token_embed = self.token_embedding_table(input_tensor) #(batch x context_size x n_embed)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) #torch.arange just creates integers from 0 to T-1; embedding this vector produces a T x C matrix (we expand the C dimension when we embed)
        x = token_embed + pos_embed #Creates a B x T x C tensor. We add the T x C information to all batches
        x = self.sa_heads(x) #feed our tensor to self-attention heads
        x = self.ffwd(x) # B,T,C
        logits = self.lm_head(x) #Creates B x T x vocab_size tensor

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

            idx_cut = input_tensor[:,-block_size:] # we need to trim down our matrix to be batch x context_length before running it through. Essentially chop of the 0th position token and add the 8th position token to maintain a block size of 8 after a generation

            logits, loss = self(idx_cut) #in pytorch when we call the __call__ method (which is what happens when we use an object as a function), we get forward(). Basically this step calls forward on our input tensor
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


    
model = BigramLanguageModel()
model_export = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    #Here is our evaluation logic. Every so often we output our loss estimates
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']}, val loss {losses['validation']}")

    #Here is the core training logic
    #
    input_batch, target_batch = get_batch('train')

    logits, loss = model(input_batch, target_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Generate 100 new tokens
print(decode(model.generate(input_tensor=torch.zeros((1,1), dtype = torch.long, device=device), max_new_tokens=500)[0].tolist()))

