import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import matplotlib.pyplot as plt   # needed to visualize loss curves
import numpy as np 
import networkx as nx
import random
import pickle
import time
import math

N = 19   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
MYN = int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)

LEARNING_RATE = 0.001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions = 500 #number of new sessions per iteration
percentile = 90 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration
			  
observation_space = 2*MYN #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
						  #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
						  #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
						  #Is there a better way to format the input to make it easier for the neural network to understand things?
lexi_order = [(i, j) for i in range(N) for j in range(i+1, N)]  # Lexicographical order 
colex_order = [(i, j) for j in range(1, N) for i in range(j)]   # Colexicographical order

						  
len_game = MYN 

INF = 1000000


# Reference blog for GNNStack and LinkPredictor: https://medium.com/stanford-cs224w/online-link-prediction-with-graph-neural-networks-46c1054f2aa4
# Reference of Main code and deep cross entropy method: https://github.com/zawagner22/cross-entropy-for-combinatorics/blob/main/demos/cem_binary_conj21.py
# Reference for reinforce_games function: https://colab.research.google.com/drive/14XYKnUoXyX8DAuaD7hnu4AxOUaQQdvWs?usp=sharing#scrollTo=pgENgiaFG40d
# The whole algorithm is inspired by the paper "Deep Cross Entropy Method for Combinatorial Optimization" by Wagner et al. (2021)

class GNNStack(torch.nn.Module):
    '''General graph neural network stack.'''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False):
        super(GNNStack, self).__init__()
        conv_model = pyg.nn.GCNConv

        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.dropout = dropout
        self.num_layers = num_layers
        self.emb = emb

        # Create num_layers GraphSAGE convs
        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers - 1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))
        
        # post-message-passing processing 
        self.post_mp = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        # Return final layer of embeddings if specified
        if self.emb:
            return x

        # Else return class probabilities (not actually used in this task, but the structure could be used for more topics related to GNNs.)
        return F.log_softmax(x, dim=1)


class LinkPredictor(nn.Module):  
    '''Link predictor. Convert node embeddings into a probability of the edge between two nodes through simple MLP.'''
    def __init__(self, in_dim, hidden_dim=32):  
        super(LinkPredictor, self).__init__()  
        # Add these learnable layers  
        self.lin1 = nn.Linear(in_dim * 2, hidden_dim)  # Combine both node features  
        self.lin2 = nn.Linear(hidden_dim, 1)   
    
    def forward(self, x_i, x_j):  
        # Concatenate the two node embeddings 
        x_i = x_i.unsqueeze(0)  # Flatten to 1d  
        x_j = x_j.unsqueeze(0) 
        x = torch.cat([x_i, x_j], dim=1)  
        # Apply non-linear transformation ReLu  
        x = F.relu(self.lin1(x))  
        # Output a probability in [0, 1]  
        x = torch.sigmoid(self.lin2(x))  
        return x 


def Wagner_to_GNN(W_vectors, N, order=lexi_order):  
    n_sessions = len(W_vectors)  
    feature_len = W_vectors.shape[1]  
    midpoint = feature_len // 2  
    
    # Split the vectors  
    state = W_vectors[:, :midpoint]  
    indicator = W_vectors[:, midpoint:]  
    
    # Initialize structures  
    edge_list = [[[], []] for _ in range(n_sessions)]  
    edge2predict = np.empty((n_sessions, 2))  
    
    # Pre-compute all (i,j) pairs in the correct order  
    ij_pairs = order
    
    for count, (i, j) in enumerate(ij_pairs):  
        # Vectorize the session loop - get all sessions where this edge exists  
        edge_sessions = np.where(state[:, count] == 1)[0]  
        predict_sessions = np.where(indicator[:, count] == 1)[0]  
        
        # Update edge lists
        for k in edge_sessions:  
            edge_list[k][0].extend([i, j])  
            edge_list[k][1].extend([j, i])  
            
        # Set next pairs of edges to predict  
        for k in predict_sessions:  
            edge2predict[k] = np.array([i, j])  
    
    # Convert to tensors 
    edge_torch_list = [torch.tensor(edge_list[k], dtype=torch.long) for k in range(n_sessions)]  
    
    return edge_torch_list, torch.tensor(edge2predict)  


def predict_edges(gnn, linkpred, states, emb, N=19):
    '''Edge prediction function'''
    edge_torch, edge2predict = Wagner_to_GNN(states, N, order=lexi_order)
    n_sessions = len(edge_torch)
    prob = torch.zeros(n_sessions)
    for k in range(n_sessions):
        x = gnn(emb.weight, edge_torch[k])
        prob[k] = linkpred(x[int(edge2predict[k,0])],x[int(edge2predict[k,1])])

    return prob

def calcScore(state, order_type=lexi_order):
	"""
	Calculates the reward for a given word. 
	This function is very slow, it can be massively sped up with numba -- but numba doesn't support networkx yet, which is very convenient to use here
	:param state: the first MYN letters of this param are the word that the neural network has constructed.


	:returns: the reward (a real number). Higher is better, the network will try to maximize this.
	"""	
	
	#Example reward function, for Conjecture 2.1
	#Given a graph, it minimizes lambda_1 + mu.
	
	#Construct the graph 
	G= nx.Graph()
	G.add_nodes_from(list(range(N)))
	count = 0
	if order_type ==lexi_order:
		for i in range(N):
			for j in range(i+1, N):
				if state[count] == 1:
					G.add_edge(i, j)
				count += 1
	else:
		for j in range(N):
			for i in range(j):
				if state[count] == 1:
					G.add_edge(i, j)
				count += 1
	
	#G is assumed to be connected in the conjecture. If it isn't, return a very negative reward.
	if not (nx.is_connected(G)):
		return -INF
		
	#Calculate the eigenvalues of G
	evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
	evalsRealAbs = np.zeros_like(evals)
	for i in range(len(evals)):
		evalsRealAbs[i] = abs(evals[i])
	lambda1 = max(evalsRealAbs)
	
	#Calculate the matching number of G
	maxMatch = nx.max_weight_matching(G)
	mu = len(maxMatch)
		
	#Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
	#We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
	myScore = math.sqrt(N-1) + 1 - lambda1 - mu
	if myScore > 0:
		#You have found a counterexample. Do something with it.
		print(state)
		# Record the counterexample as a file for further usage.
		with open ('counterexample.txt', 'a') as f:
			f.write(str(state) + '\n')
		nx.draw_kamada_kawai(G)
		plt.show()
		exit()
		
	return myScore


def generate_session(gnn, linkpred, n_sessions, emb, verbose = 1):
	"""
	Play n_session games using agent neural network.
	Terminate when games finish 
	
	Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	"""
	states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
	actions = np.zeros([n_sessions, len_game], dtype = int)
	state_next = np.zeros([n_sessions,observation_space], dtype = int)
	prob = np.zeros(n_sessions)
	states[:,MYN,0] = 1
	step = 0
	total_score = np.zeros([n_sessions])
	recordsess_time = 0
	play_time = 0
	scorecalc_time = 0
	pred_time = 0
	while (True):
		step += 1		
		tic = time.time()
		prob = predict_edges(gnn, linkpred, states[:,:,step-1], emb) # Main change in this function
		pred_time += time.time()-tic
		
		for i in range(n_sessions):
			
			if np.random.rand() < prob[i]:
				action = 1
			else:
				action = 0
			actions[i][step-1] = action
			tic = time.time()
			state_next[i] = states[i,:,step-1]
			play_time += time.time()-tic
			if (action > 0):
				state_next[i][step-1] = action		
			state_next[i][MYN + step-1] = 0
			if (step < MYN):
				state_next[i][MYN + step] = 1			
			terminal = step == MYN
			tic = time.time()
			if terminal:
				total_score[i] = calcScore(state_next[i])
			scorecalc_time += time.time()-tic
			tic = time.time()
			if not terminal:
				states[i,:,step] = state_next[i]			
			recordsess_time += time.time()-tic
			
		
		if terminal:
			break
	#If you want, print out how much time each step has taken. This is useful to find the bottleneck in the program.		
	if (verbose):
		print("Predict: "+str(pred_time)+", play: " + str(play_time) +", scorecalc: " + str(scorecalc_time) +", recordsess: " + str(recordsess_time))
	return states, actions, total_score


def reinforce_games(  
    gnn, 
    linkpred,
    emb,
    optimiser,  
    states,  
    actions,  
    batch_size=MYN,  
):  
    """  
    Given sessions of elites, reinforce each move in each session.  
    Modified to be compatible with GNN-based models.  
    """  
    # Put model in training mode  
    gnn.train() 
    linkpred.train()
    
    
    # Convert to PyTorch tensors if they aren't already  
    if not isinstance(states, torch.Tensor):  
        states = torch.tensor(states, dtype=torch.float32)  
    if not isinstance(actions, torch.Tensor):  
        actions = torch.tensor(actions, dtype=torch.float32)  
    
    # Random permutation for batch training  
    n_samples = states.shape[0]  
    shuffle = torch.randperm(n_samples)  
    
    # Loss function  
    criterion = nn.BCELoss()  
    
    # Track total loss  
    total_loss = 0  
    num_batches = 0  
    
    # Train in batches  
    for i in range(0, n_samples, batch_size):  
        batch_indices = shuffle[i:i+batch_size] 

        # Skip if batch is empty  
        if len(batch_indices) == 0:  
            continue 

        optimiser.zero_grad()  
        batch_states = states[batch_indices]  
        
        # Get predictions as tensor with gradients  
        predicted = predict_edges(gnn, linkpred, batch_states, emb) 
        
        # Ensure predictions are tensor  
        if not isinstance(predicted, torch.Tensor):  
            predicted = torch.tensor(predicted, dtype=torch.float32, requires_grad=True)  
        # Ensure batch_actions has the right shape  
        batch_actions = actions[batch_indices]  
        
        # Compute loss  
        loss = criterion(predicted, batch_actions)  
        
        # Backpropagate and update weights  
        loss.backward()  
        optimiser.step()   
        total_loss += loss.item()  
        num_batches += 1  
    
    # Set model back to evaluation mode  
    gnn.eval()  
    linkpred.eval()
    # Return average loss  
    return total_loss / num_batches, emb 


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
	"""
	Select states and actions from games that have rewards >= percentile
	:param states_batch: list of lists of states, states_batch[session_i][t]
	:param actions_batch: list of lists of actions, actions_batch[session_i][t]
	:param rewards_batch: list of rewards, rewards_batch[session_i]

	:returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
	
	This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	elite_states = []
	elite_actions = []
	elite_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:		
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				for item in states_batch[i]:
					elite_states.append(item.tolist())
				for item in actions_batch[i]:
					elite_actions.append(item)			
			counter -= 1
	elite_states = np.array(elite_states, dtype = int)	
	elite_actions = np.array(elite_actions, dtype = int)	
	return elite_states, elite_actions
	
def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
	"""
	Select all the sessions that will survive to the next generation
	Similar to select_elites function
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	super_states = []
	super_actions = []
	super_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				super_states.append(states_batch[i])
				super_actions.append(actions_batch[i])
				super_rewards.append(rewards_batch[i])
				counter -= 1
	super_states = np.array(super_states, dtype = int)
	super_actions = np.array(super_actions, dtype = int)
	super_rewards = np.array(super_rewards)
	return super_states, super_actions, super_rewards



gnn = GNNStack(input_dim=64, hidden_dim=128, output_dim=64,num_layers=4, dropout=0.1)
linkpred = LinkPredictor(in_dim=64)
emb = torch.nn.Embedding(N,64)

super_states =  np.empty((0,len_game,observation_space), dtype = int)
super_actions = np.array([], dtype = int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0


opt = torch.optim.Adam(list(gnn.parameters())+list(linkpred.parameters())+list(emb.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)  
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5)  


myRand = random.randint(0,1000) #used in the filename

for i in range(10000): 
	#generate new sessions
	tic = time.time()
	sessions = generate_session(gnn,linkpred,n_sessions,emb,0) #change 0 to 1 to print out how much time each step in generate_session takes 
	sessgen_time = time.time()-tic
	tic = time.time()
	
	states_batch = np.array(sessions[0], dtype = int)
	actions_batch = np.array(sessions[1], dtype = int)
	rewards_batch = np.array(sessions[2])
	states_batch = np.transpose(states_batch,axes=[0,2,1])
	
	states_batch = np.append(states_batch,super_states,axis=0)

	if i>0:
		actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)	
	rewards_batch = np.append(rewards_batch,super_rewards)
		
	randomcomp_time = time.time()-tic 
	tic = time.time()

	elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
	select1_time = time.time()-tic

	tic = time.time()
	super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
	select2_time = time.time()-tic
	
	tic = time.time()
	super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
	super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
	select3_time = time.time()-tic
	
	tic = time.time()
	
	loss, emb = reinforce_games(  
        gnn,
		linkpred,  
		emb,
        optimiser=opt,  
        states=elite_states,  
        actions=elite_actions,  
        batch_size=MYN,  # Adjusted batch size to be more flexible  
    )
	print(loss)
	scheduler.step(loss)  # Adjust learning rate based on loss  
	fit_time = time.time()-tic
	
	tic = time.time()
	
	super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
	super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
	super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
	
	rewards_batch_copy = rewards_batch.copy()
	rewards_batch.sort()
	mean_all_reward = np.mean(rewards_batch[-100:])	
	mean_best_reward = np.mean(super_rewards)	

	score_time = time.time()-tic
	
	print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
	
	#uncomment below line to print out how much time each step in this loop takes. 
	print(	"Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
	
	with open(f'reinforce_loss_{myRand}.txt', 'a') as f:  
		f.write(f"{i},{loss}\n")  

	with open(f'best_species_pickle_{myRand}.txt', 'wb') as fp:  
		pickle.dump(super_actions, fp)  

	with open(f'best_species_txt_{myRand}.txt', 'a') as f:  
		f.write(f"Iteration {i}: {super_actions[0]}\n") 

	with open(f'best_species_rewards_{myRand}.txt', 'a') as f:  
		f.write(f"Iteration {i}: {super_rewards[0]}\n")  

	with open(f'best_10_percent_rewards_{myRand}.txt', 'a') as f:  
		f.write(str(mean_all_reward)+"\n")  

	with open(f'best_elite_rewards_{myRand}.txt', 'a') as f:  
		f.write(str(mean_best_reward)+"\n")  

	# Find and store worst actions and rewards  
	worst_reward_idx = np.argmin(rewards_batch_copy)  
	worst_reward = rewards_batch_copy[worst_reward_idx]  
	worst_action = actions_batch[worst_reward_idx] 

	with open(f'worst_reward_{myRand}.txt', 'a') as f:  
		f.write(f"Iteration {i}: {worst_reward}\n")  

	with open(f'worst_actions_{myRand}.txt', 'a') as f:  
		f.write(f"Iteration {i}: {worst_action}\n")  