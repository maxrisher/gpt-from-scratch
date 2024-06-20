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