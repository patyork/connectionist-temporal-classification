import numpy as np
from itertools import groupby

class RecursiveCTCLayer:
    def __init__(self, alphabet=np.arange(29), remove_duplicates=True):
        self.A = alphabet
        self.blank = len(alphabet)
        self.inputs = None
        self.sequence = None
        self.sequence_prime = None
        self.matrixf = None
        self.matrixb = None

        self.T = None
        self.U = None

        self.remove_duplicates=remove_duplicates

    def recurrence_relationship(self, size):
        big_I = np.eye(size+2)
        return np.eye(size) + big_I[2:, 1:-1] + big_I[2:, :-2] * (np.arange(size) % 2)

    # Remove consecutive symbols and blanks
    def F(self, pi):
        return [a for a in [key for key, _ in groupby(pi)] if a != self.blank]

    # Insert blanks between unique symbols, and at the beginning and end
    def make_l_prime(self, l):
        result = [self.blank] * (len(l) * 2 + 1)
        result[1::2] = l
        return result
        # return [blank] + sum([[i, blank] for i in l], [])

    # Calculate p(sequence|inputs)
    def ctc(self, inputs, sequence):
        self.inputs = inputs
        self.T = len(inputs)
        self.sequence = sequence
        if self.remove_duplicates:
            self.sequence_prime = self.make_l_prime(self.F(sequence))
        else:
            self.sequence_prime = self.make_l_prime(sequence)
        self.U = len(self.sequence_prime)
        self.matrixf = np.zeros((self.T, self.T))
        self.matrixb = np.zeros((self.T, self.T))



        fp = self.forward(self.T-1, self.U-1)
        #print 'matf\n', self.matrixf, 'endmatf'
        self.matrixf = np.zeros((self.T, self.T))
        return fp + self.forward(self.T-1, self.U-2)

        summation = 0.0
        for s in np.arange(self.U):
            self.matrixf = np.zeros((self.T, self.T))
            self.matrixb = np.zeros((self.T, self.T))

            summation += self.forward(self.T-1, s) * self.backward(self.T-1, s) / y[self.T-1][self.sequence_prime[s]]
        return fp

    # DP (recursive) as described by the paper
    def forward(self, t, u):
        if self.matrixf[t][u] != 0:
            return self.matrixf[t][u]

        if t==0 and u==0:
            prob = self.inputs[0][self.blank]
            self.matrixf[t][u] = prob
            return prob
        elif t==0 and u==1:
            prob = self.inputs[0][self.sequence[0]]
            self.matrixf[t][u] = prob
            return prob
        elif t==0: return 0
        elif u<1 or u<(len(self.sequence_prime) - 2*(self.T-1 - t)-1): return 0

        if self.sequence_prime[u]==self.blank or self.sequence_prime[u-2]==self.sequence_prime[u]:
            prob = (self.forward(t-1, u) +
                    self.forward(t-1, u-1)) *\
                    self.inputs[t][self.sequence_prime[u]]
            self.matrixf[t][u] = prob
            return prob
        else:
            prob = (self.forward(t-1, u) +
                    self.forward(t-1, u-1) +
                    self.forward(t-1, u-2)) *\
                    self.inputs[t][self.sequence_prime[u]]
            self.matrixf[t][u] = prob
            return prob

    def backward(self, t, u):
        if self.matrixb[t][u] != 0:
            self.matrixb[t][u]

        if t==self.T-1 and u==len(self.sequence_prime)-1:
            prob = self.inputs[self.T-1][self.blank]
            self.matrixb[t][u] = prob
            return prob
        elif t==self.T-1 and u==len(self.sequence_prime)-2:
            prob = self.inputs[self.T-1][self.sequence[-1]]
            self.matrixb[t][u] = prob
            return prob
        elif t==self.T-1:
            return 0
        elif u>2*t-1 or u>len(self.sequence_prime)-1:
            return 0

        if self.sequence_prime[u]==self.blank:
            prob = (self.backward(t+1, u) +
                    self.backward(t+1, u+1)) *\
                self.inputs[t][self.sequence_prime[u]]
            self.matrixb[t][u] = prob
            return prob

        # this is almost certainly incorrect, but there is an out-of-bounds error without it that I cannot track down
        if u==len(self.sequence_prime)-2: return 0

        elif self.sequence_prime[u+2]==self.sequence_prime[u]:
            prob = (self.backward(t+1, u) +
                    self.backward(t+1, u+1)) *\
                self.inputs[t][self.sequence_prime[u]]
            self.matrixb[t][u] = prob
            return prob
        else:
            prob = (self.backward(t+1, u) +
                    self.backward(t+1, u+1) +
                    self.backward(t+1, u+2)) *\
                self.inputs[t][self.sequence_prime[u]]
            self.matrixb[t][u] = prob
            return prob

    def alpha_beta(self, inputs, sequence):
        self.inputs = inputs
        self.T = len(inputs)
        self.sequence = sequence
        self.sequence_prime = self.make_l_prime(self.F(sequence))
        self.U = len(self.sequence_prime)
        self.matrixf = np.zeros((self.T, self.T))
        self.matrixb = np.zeros((self.T, self.T))

        alpha_beta = []

        for t in np.arange(self.T):
            ab_t = []
            for u in np.arange(self.U):
                self.matrixf = np.zeros((self.T, self.T))
                self.matrixb = np.zeros((self.T, self.T))
                _f = self.forward(t, u)
                _b = self.backward(t, u)
                ab_t.append(_f * _b)
            alpha_beta.append(ab_t)

        return np.asarray(alpha_beta, dtype=float)