import torch

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)

# double parentheses because we need to send a tuple, not three separate arguments to torch.zeros()
x_bag_of_words = torch.zeros((B,T,C))
for b in range (B):
    for t in range(T):
        # pick out batch b. Get the vectors for all time up to and including t.
        x_prev = x[b,:t+1] # will be of shape t x C
        #find the vector in the x_bag_of_words tensor at b, t. give it the average of the time channels up to and including that time
        #average over the first dimension of x_prev (ie. for each channel, average over all time; NOT for each time average over all channels)
        x_bag_of_words[b,t] = torch.mean(x_prev, 0) 

print(x_bag_of_words)

# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)