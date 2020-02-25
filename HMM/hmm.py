from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        self.alpha = None
        self.beta = None
        self.PO = None
        self.gamma = None
        self.psi = None
        self.path = None

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        for t in range(L):
            for s in range(S):
                if(t == 0):
                    alpha[s, t] = self.pi[s] * self.B[s, self.obs_dict[Osequence[t]]]
                else:
                    sum = 0
                    for sp in range(S):
                        sum += self.A[sp, s] * alpha[sp, t - 1]
                    alpha[s, t] = self.B[s, self.obs_dict[Osequence[t]]] * sum
        self.alpha = alpha
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        for t in list(range(L - 1, -1, -1)):
            for s in range(S):
                if(t == L - 1):
                    beta[s, t] = 1
                else:
                    beta[s, t] = 0
                    for sp in range(S):
                        beta[s, t] += self.A[s, sp] * self.B[sp, self.obs_dict[Osequence[t + 1]]] * beta[sp, t + 1]
        self.beta = beta
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        for s in range(np.size(self.alpha, 0)):
            prob += self.alpha[s, np.size(self.alpha, 1) - 1]
        self.PO = prob
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        for i in range(S):
            for t in range(L):
                prob[i, t] = self.alpha[i, t] * self.beta[i, t] / self.PO
        self.gamma = prob
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        for s in range(S):
            for sp in range(S):
                for t in range(L - 1):
                    prob[s, sp, t] = self.alpha[s, t] * self.A[s, sp] * self.B[sp, self.obs_dict[Osequence[t + 1]]] * self.beta[sp, t + 1] / self.PO
        self.psi = prob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        L = len(Osequence)
        S = len(self.pi)
        delta = np.zeros([S, L])
        Delta = (np.zeros([S, L])).astype(int)
        for s in range(S):
            delta[s, 0] = self.pi[s] * self.B[s, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            for s in range(S):
                maxx = 0
                amax = 0
                for sp in range(S):
                    if(maxx < self.A[sp, s] * delta[sp, t - 1]):
                        maxx = self.A[sp, s] * delta[sp, t - 1]
                        amax = sp
                delta[s, t] = self.B[s, self.obs_dict[Osequence[t]]] * maxx
                Delta[s, t] = int(amax)
        maxx = 0
        amax = 0
        for s in range(S):
            if(maxx < delta[s, L - 1]):
                maxx = delta[s, L - 1]
                amax = s
        idx_of_path = (np.zeros(L)).astype(int)
        idx_of_path[L - 1] = int(amax)
        for t in range(L - 1, 0, -1):
            idx_of_path[t - 1] = Delta[idx_of_path[t], t]
        inverted_dict = dict([[v, k] for k, v in self.state_dict.items()])
        for t in range(L):
            path.append(inverted_dict[idx_of_path[t]])
        self.path = path
        ###################################################
        return path


