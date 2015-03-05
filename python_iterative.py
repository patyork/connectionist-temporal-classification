import numpy as np
from itertools import groupby

class IterativeCTCLayer:
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


    # This function acts as a sanity check:
    #      P(sequence|inputs) = forward(T, U) + forward(T, U-1)
    #      P(sequence|inputs) = forward(T, U)*backward(T, U)/inputs[T][U]
    #
    # therefore
    #
    #      forward(T, U) + forward(T, U-1) == forward(T, U)*backward(T, U)/inputs[T][U]
    #
    def ctc_check(self, inputs, sequence):
        self.inputs = inputs
        self.T = len(inputs)
        self.sequence = sequence
        self.remove_duplicates = True
        if self.remove_duplicates:
            self.sequence_prime = self.make_l_prime(self.F(self.sequence))
        else:
            self.sequence_prime = self.make_l_prime(self.sequence)
        self.U = len(self.sequence_prime)
        self.matrixf = np.zeros((self.T, self.T))
        self.matrixb = np.zeros((self.T, self.T))

        # forward(T, U)
        self.calc_mat_f(self.T-1, self.U-1)
        self.forward_wrapper()
        fp1 = np.sum(self.matrixf[-1, :])

        # backward(T, U)
        self.calc_mat_b(self.T-1, self.U-1)
        self.backward_wrapper()
        bp1 = np.sum(self.matrixb[-1,:])

        print fp1, bp1, self.inputs[-1][-1]
        print self.matrixb
        print self.matrixf

    # Calculate p(sequence|inputs)
    def ctc(self, inputs, sequence):
        self.inputs = inputs
        self.T = len(inputs)
        self.sequence = sequence
        self.sequence_prime = sequence
        self.remove_duplicates = True
        if self.remove_duplicates:
            self.sequence_prime = self.make_l_prime(self.F(self.sequence))
        else:
            self.sequence_prime = self.make_l_prime(self.sequence)
        self.U = len(self.sequence_prime)
        self.matrixf = np.zeros((self.T, self.T))
        self.matrixb = np.zeros((self.T, self.T))


        self.calc_mat_f(self.T-1, self.U-1)     # Calculate the DP matrix of t,u values necessary into self.matrixf
        self.forward_wrapper()                  # Overwrite self.matrixf with the forward computations
        fp = np.sum(self.matrixf[-1,:])        # The sum of the bottom row is alpha(T, U)

        self.calc_mat_b(self.T-1, self.U-1)
        self.backward_wrapper()
        bp = np.sum(self.matrixb[-1,:])

        # alpha(T, U) * beta(T, U) / inputs[T][U]
        return fp * bp / self.inputs[-1][-1]

    # Calculate the necessary forward probabilities iteratively
    def forward_wrapper(self):
        for t in np.arange(np.shape(self.matrixf)[0]):
            for u in np.arange(np.shape(self.matrixf)[1]):
                self.matrixf[t][u] = self.forward(t, u)

    # Calculate the necessary backward probabilities iteratively
    def backward_wrapper(self):
        for t in np.arange(np.shape(self.matrixb)[0]-1, -1, -1):
            for u in np.arange(np.shape(self.matrixb)[1]-1, -1, -1):
                self.matrixb[t][u] = self.backward(t, u)


    # Calculate a matrix that determines which pairs of (t, u) are required for the forward calculation
    # This follows the recursive algorithm defined by the paper, but with a much lower cost per recursion level
    # The actual probabilities at each t/u step will be calculated iteratively using this matrix as a guide
    def calc_mat_f(self, t, u):
        if self.matrixf[t][u] != 0:
            return
        if t==0 and u==0:
            self.matrixf[t][u] = 1
            return
        elif t==0 and u==1:
            self.matrixf[t][u] = 1
            return
        elif t==0:
            return
        elif u<1 or u<(len(self.sequence_prime) - 2*(self.T-1 - t)-1):
            return

        if self.sequence_prime[u]==self.blank or self.sequence_prime[u-2]==self.sequence_prime[u]:
            self.calc_mat_f(t-1, u)
            self.calc_mat_f(t-1, u-1)
            self.matrixf[t][u] = 1
            return
        else:
            self.calc_mat_f(t-1, u)
            self.calc_mat_f(t-1, u-1)
            self.calc_mat_f(t-1, u-2)
            self.matrixf[t][u] = 1
            return

    def calc_mat_b(self, t, u):
        if self.matrixb[t][u] != 0:
            return
        elif t==self.T-1 and (u==len(self.sequence_prime)-1 or u==len(self.sequence_prime)-2):
            self.matrixb[t][u] = 1
            return
        elif t==self.T-1:
            return
        elif u+1>2*(t+1) or u>len(self.sequence_prime)-1:
            return
        elif self.sequence_prime[u]==self.blank:
            self.calc_mat_b(t+1, u)
            self.calc_mat_b(t+1, u+1)
            self.matrixb[t][u] = 1
            return

        # this is almost certainly incorrect, but there is an out-of-bounds error without it that I cannot track down
        elif u==len(self.sequence_prime)-2:
            return

        elif self.sequence_prime[u+2]==self.sequence_prime[u]:
            self.calc_mat_b(t+1, u)
            self.calc_mat_b(t+1, u+1)
            self.matrixb[t][u] = 1
            return
        else:
            self.calc_mat_b(t+1, u)
            self.calc_mat_b(t+1, u+1)
            self.calc_mat_b(t+1, u+2)
            self.matrixb[t][u] = 1
            return

    # Iterative approach to the forward algorithm
    def forward(self, t, u):
        if self.matrixf[t][u] != 1:
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
            prob = (self.matrixf[t-1][u] +
                    self.matrixf[t-1][u-1]) *\
                    self.inputs[t][self.sequence_prime[u]]
            self.matrixf[t][u] = prob
            return prob
        else:
            prob = (self.matrixf[t-1][u] +
                    self.matrixf[t-1][u-1] +
                    self.matrixf[t-1][u-2]) *\
                    self.inputs[t][self.sequence_prime[u]]
            self.matrixf[t][u] = prob
            return prob

    # Iterative approach to the backward algorithm
    def backward(self, t, u):
        if self.matrixb[t][u] != 1:
            return self.matrixb[t][u]

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
        elif u+1>2*(t+1) or u>len(self.sequence_prime)-1:
            return 0

        if self.sequence_prime[u]==self.blank:
            prob = (self.matrixb[t+1][u] +
                    self.matrixb[t+1][u+1]) *\
                self.inputs[t][self.sequence_prime[u]]
            self.matrixb[t][u] = prob
            return prob

        # this is almost certainly incorrect, but there is an out-of-bounds error without it that I cannot track down
        if u==len(self.sequence_prime)-2: return 0

        elif self.sequence_prime[u+2]==self.sequence_prime[u]:
            prob = (self.matrixb[t+1][u] +
                    self.matrixb[t+1][u+1]) *\
                self.inputs[t][self.sequence_prime[u]]
            self.matrixb[t][u] = prob
            return prob
        else:
            prob = (self.matrixb[t+1][u] +
                    self.matrixb[t+1][u+1] +
                    self.matrixb[t+1][u+2]) *\
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