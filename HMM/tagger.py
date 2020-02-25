import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	states = set(tags)
	num_of_sentences = len(train_data)
	idx = 0
	num_of_unique_states = len(states)
	state_dict = dict()
	for state in states:
		state_dict[state] = idx
		idx += 1
	assert idx == num_of_unique_states
	assert num_of_unique_states == len(state_dict)
	observations = set()
	for sentence in train_data:
		for word in sentence.words:
			observations.add(word)
	num_of_unique_obs = len(observations)
	obs_dict = dict()
	idx = 0
	for obsv in observations:
		obs_dict[obsv] = idx
		idx += 1
	assert idx == num_of_unique_obs
	pi = np.zeros(num_of_unique_states)
	count_of_possible_starters = dict()
	for sentence in train_data:
		if sentence.tags[0] not in count_of_possible_starters:
			count_of_possible_starters[sentence.tags[0]] = 1
		else:
			count_of_possible_starters[sentence.tags[0]] += 1
	for state, count in count_of_possible_starters.items():
		idx = state_dict[state]
		assert idx < num_of_unique_states
		pi[idx] = count/num_of_sentences
	A = np.zeros([num_of_unique_states, num_of_unique_states])
	master_list_of_tuples_states = []
	for sentence in train_data:
		list_of_tuples = [(sentence.tags[idx], sentence.tags[idx + 1]) for idx in range(sentence.length - 1)]
		master_list_of_tuples_states.append(list_of_tuples)
	master_list_of_tuples_indexes = []
	for sbl in master_list_of_tuples_states:
		for aa, bb in sbl:
			master_list_of_tuples_indexes.append((state_dict[aa], state_dict[bb]))
	for i in range(num_of_unique_states):
		dr = 0.0
		for cc, dd in master_list_of_tuples_indexes:
			if (i == cc):
				dr += 1
				A[cc, dd] += 1
		if dr:
			A[i, :] = (1/dr) * A[i, :]
	state_obs_pairs_idx = []
	for sentence in train_data:
		for i in range(sentence.length):
			state_obs_pairs_idx.append((state_dict[sentence.tags[i]], obs_dict[sentence.words[i]]))
	B = np.zeros([num_of_unique_states, num_of_unique_obs])
	for i in range(num_of_unique_states):
		dr = 0.0
		for cc, dd in state_obs_pairs_idx:
			if (i == cc):
				dr += 1
				B[cc, dd] += 1
		if dr:
			B[i, :] = (1 / dr) * B[i, :]
	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	S, NO = model.B.shape
	next_idx = NO
	C = np.full((S, 1), 1e-6)
	for sentence in test_data:
		for word in sentence.words:
			if word not in model.obs_dict:
				model.obs_dict[word] = next_idx
				next_idx += 1
				model.B = np.concatenate((model.B, C), axis=1)
	assert next_idx == model.B.shape[1]
	for sentence in test_data:
		tagging.append(model.viterbi(sentence.words))
	###################################################
	return tagging

