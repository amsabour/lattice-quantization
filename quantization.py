import numpy as np
import random
import math
import torch



class ModuloCompressor:
    def __init__(self, q):
        self.q = q
        self.length = int(np.ceil(np.log2(q) / 8))
        
    def compress(self, a):
        l = []
        for i in range(self.length):
            l.append(torch.remainder(a >> (8 * i), 256).type(torch.uint8))
        new_a = torch.cat(l)
        return new_a

class ModuloDecompressor:
    def __init__(self, q):
        self.q = q
        self.length = int(np.ceil(np.log2(q) / 8))
    
    def decompress(self, a):
        a = a.flatten()
        size = a.shape[0]
        assert size % self.length == 0
        real_size = size // self.length
        
        result = torch.zeros(real_size).cuda()

        for i in range(self.length):    
            result[:] +=  a[i * real_size : (i + 1) * real_size].type(result.dtype) << (8 *  i)
        
        return result

    
    
class Encoder:
    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
    
    def encode_bucket(self, a):
        raise NotImplementedError
    
    def encode(self, a):
        quantized = torch.empty_like(a)
    
        for i in range((len(a) + self.bucket_size - 1) // self.bucket_size):
            quantized[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))] = self.encode_bucket(a[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))])
            
        return quantized
    
    def encode_to(self, a, dest):
        for i in range((len(a) + self.bucket_size - 1) // self.bucket_size):
            dest[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))] = self.encode_bucket(a[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))])

class Decoder:
    def __init__(self, bucket_size):
        self.bucket_size = bucket_size
    
    def decode_bucket(self, quantized_vector, b):
        raise NotImplementedError
        
    def decode(self, a, b):
        decoded = torch.empty_like(a)
        
        for i in range((len(a) + self.bucket_size - 1) // self.bucket_size):
            decoded[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))] = self.decode_bucket(a[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))], b[i * self.bucket_size:min((i + 1) * self.bucket_size, len(b))])
            
        return decoded
    
    def decode_to(self, a, b, dest):
        for i in range((len(a) + self.bucket_size - 1) // self.bucket_size):
            dest[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))] = self.decode_bucket(a[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))], b[i * self.bucket_size:min((i + 1) * self.bucket_size, len(b))])

            
            
class SimpleLatticeEncoder(Encoder):
    def __init__(self, bucket_size, n):

        self.d = bucket_size  # Dimension
        self.sigma = 0.0035  # standard deviation (estimated from the gradient)
        self.n = n  # number of nodes
        self.q = self.d ** 2  # quantization level
        self.k = math.floor(3 / 4 * self.d)
        self.epsilon = self.sigma / self.n

        super(SimpleLatticeEncoder, self).__init__(self.d)
    
    
    def encode_bucket(self, a):
        input_vec = a
        
        scaled_input = input_vec / self.epsilon # devide by the epsilon
        scaled_input = torch.round(scaled_input).type(torch.int32).cuda() # make it integer
        encoded_vector = torch.remainder(scaled_input, self.q).cuda() # mod q
        
        return encoded_vector
    
    def encode(self, a):
        return self.encode_bucket(a)


class SimpleLatticeDecoder(Decoder):
    def __init__(self, bucket_size, n):

        self.d = bucket_size  # Dimension
        self.sigma = 0.0035  # standard deviation (estimated from the gradient)
        self.n = n  # number of nodes
        self.q = self.d ** 2  # quantization level
        self.k = math.floor(3 / 4 * self.d)
        self.epsilon = self.sigma / self.n

        super(SimpleLatticeDecoder, self).__init__(self.d)
    
    
    def decode_bucket(self, quantized_vector, b):
        # Decoding phase:
        scaled_qv = quantized_vector * self.epsilon #multiply by epsilon
        scaled_qv = b - scaled_qv # subtract the result from the other estimator
        decoded_vec = torch.round(scaled_qv / (self.q * self.epsilon)).cuda() * self.q * self.epsilon #divide by q*epsilon, round and multiply by q*epsilon again
        decoded_vec = decoded_vec + (quantized_vector * self.epsilon) # add the quantized_vector*epsilon to the resolt

        return decoded_vec
    
    def decode(self, a, b):
        return self.decode_bucket(a, b)
            