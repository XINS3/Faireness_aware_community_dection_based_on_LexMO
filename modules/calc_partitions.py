import math
from collections import defaultdict

from networkx.utils import py_random_state

from .helpers import (_full_partition_colors, diversity_fairness_gain,
                      modularity_fairness_gain, neighbor_weights)


# Calculate one level of partitions based on modularity alone
@py_random_state("seed")
def _calculate_partition_mod(G, n, m, partition, colors, color_dist, phi, resolution=1, is_directed=False, seed=None, mode="base"):
	# At start, assign each node to a community 
	comms = {u: i for i,u in enumerate(G.nodes())}
	inner_partition = [{u} for u in G.nodes()]

	color_list=color_dist.keys()
	K_cols=len(color_list)

	original_partition=partition.copy()

	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	# Get all neighbours, including R/B
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
	
	# Get sum of edge weights for all B/R nodes
	sum_all = {u: len(nbrs[u]) for u in G}

	degrees=dict(G.degree(weight="weight"))
	Stot=list(degrees.values())

	# Do random shuffle on seed
	rand_nodes=list(G.nodes)
	seed.shuffle(rand_nodes)


	# Start moving nodes here
	n_moves=1
	improvement=False

	while n_moves>0:
		n_moves=0
		for u in rand_nodes:
			mod_best=0
			comm_best=comms[u]

			# Get degrees
			degree=degrees[u]
			Stot[comm_best]-=degree

			# For each node, calculate the weights of its neighbours in the same comm
			w2c=neighbor_weights(nbrs[u],comms)

			# Calculate modularity remove cost
			mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)

			# For each neighbour of u: 
			for nbr_comm, wt in w2c.items():
				# Calculate modularity gain
				mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)
				# If improves over previous best, assign as best.
				if mod_gain > mod_best:
					# # @DEBUG
					# print(f"Move {u}->{nbr_comm} found:")
					# print(f"mod_best={mod_best}, mod_gain={mod_gain}")

					mod_best = mod_gain
					comm_best = nbr_comm

			# Then finalize move if necessary
			Stot[comm_best]+=degree
			if comm_best != comms[u]:
				# Finalize move

				# Get nodes in com
				com = G.nodes[u].get("nodes", {u})

				# Update partition, remove com nodes from comms(u)
				partition[comms[u]].difference_update(com)
				# Update inner partition, remove u from comms(u)
				inner_partition[comms[u]].remove(u)
				# Update partition, add com nodes to comm_best
				partition[comm_best].update(com)
				# Update inner partition, add u to comm_best
				inner_partition[comm_best].add(u)
				# Signal improvement
				improvement = True
					
				# Comment for infinite loop on a=0
				n_moves += 1

				# Change new best community for u
				comms[u] = comm_best

	# Filter out all empty partitions
	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))


	# Now calculate all colours for generated partitioning
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# For all nodes u in ci, check sums of colors
			sum_cols=[0 for _c in color_list]
			for u in comm:
				# Extend for multiple colors
				for col_ind,col in enumerate(color_list):
					if colors[u]==col:
						sum_cols[col_ind]+=1
				
			min_balance=1.0
			# Iterate over all colors to find min balance for community
			for col_ind,col in enumerate(color_list):
				sum_color=sum_cols[col_ind]

				# If any sum==0, or the sum of the color==len(ci): Leave balance to 0
				if sum_color==0 or sum_color==n_ci: 
					min_balance=0.0
					break

				# Otherwise: find if balance is min
				bal_score=sum_color/(n_ci-sum_color)
				if bal_score<min_balance:
					min_balance=bal_score

			# Set min_balance as the score. Normalize by comm size
			balance_ci=(K_cols-1)*min_balance

			# Originally set fscore as balance_ci
			fscore_ci=balance_ci

			# Change score to fexp calculation if mode is fexp
			if mode=="fexp":
				# First, calculate extra nodes for the community
				sum_dist=0
				for col in color_list:
					sum_dist+=math.floor(color_dist[col]*n_ci/n)
				n_extra=n_ci-sum_dist

				# Calculate F_exp(c_i)
				f_exp=(K_cols*phi*n_ci+(phi+K_cols-1-(phi*K_cols))*n_extra) /((K_cols*n_ci + (phi-1)*n_extra))

				# Get final score
				if n_ci>=K_cols:
					fscore_ci=min(1.0,1.0-(f_exp-balance_ci))
				else:
					fscore_ci=0.0

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)

	return partition, inner_partition, improvement, partition_colors_new



# Calculate one level of partitions on full Obj=a*Q+(1-a)*F
@py_random_state("seed")
def _calculate_partition_obj(G, n, m, partition, colors, color_dist, partition_cols, phi, alpha=0.9, resolution=1, threshold=0.0000001, is_directed=False, seed=None):
	# At start, assign each node to a community 
	comms = {u: i for i,u in enumerate(G.nodes())}
	inner_partition = [{u} for u in G.nodes()]

	color_list=color_dist.keys()
	K_cols=len(color_list)

	partition_colors=partition_cols.copy()
	original_partition=partition.copy()

	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	# Get all neighbours, including R/B
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
	
	# Get sum of edge weights for all B/R nodes
	sum_all = {u: len(nbrs[u]) for u in G}

	degrees=dict(G.degree(weight="weight"))
	Stot=list(degrees.values())


	# Do random shuffle on seed
	rand_nodes=list(G.nodes)
	seed.shuffle(rand_nodes)


	# Start calculating movements to other communities
	n_moves=1
	improvement=False

	while n_moves>0:
		n_moves=0
			
		# For each node:
		for u in rand_nodes:
			## Start with opt score for its current community?
			#  Idea: the minimum gain score is 0 for modularity and the previous fair
			#     score for the current community =>a*0+(1-a)*curr score
			opt_best=0.0
			#opt_best=(1-alpha)*partition_colors[comms[u]]["score"]
			comm_best=comms[u]
			new_fair_update=0.0
			post_fair_update=0.0

			# Get degrees
			degree=degrees[u]
			Stot[comm_best]-=degree

			# For each node, calculate the weights of its neighbours in the same comm
			w2c=neighbor_weights(nbrs[u],comms)

			
			# Calculate modularity remove cost
			mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)

			# Calculate fairness (balance) current cost
			#### @TODO: slowest part, probably. Can be optimized?
			com = G.nodes[u].get("nodes", {u})


			# For nodes under u: calculate their color scores
			curr_colors=[0 for _col in color_list]
			for i in com:
				for col_ind, col in enumerate(color_list):
					if colors[i]==col:
						curr_colors[col_ind]+=1

			# Curr fair: all nodes in comms(u)
			all_colors=[partition_colors[comms[u]][col] for col in color_list]			
	
			min_fair=1.0
			sum_pnodes=0
			for col_ind, col in enumerate(color_list):
				sum_color=all_colors[col_ind]
				sum_pnodes+=sum_color
				# Min=0 if any color is 0
				if sum_color==0: 
					min_fair=0.0
				else:
					up_score=0.0
					if len(com)>sum_color:
						# See if minimum color score is less than current min
						up_score=sum_color/(len(com)-sum_color)

					# Set to min if less than current
					if up_score<min_fair:
						min_fair=up_score

			# normalize by community size ratio
			curr_fair=(K_cols-1)*min_fair*len(com)/n

			# Calculate post_fair score of current community after moving u
			post_colors=[all_colors[col_ind]-curr_colors[col_ind] for col_ind,_col in enumerate(color_list)]
			
			# Important: add failsafe here. If post drops <0, do not allow move. cont
			drop_below_flag=False
			for col_ind,_col in enumerate(color_list):
				if post_colors[col_ind]<0: 
					drop_below_flag=True
					break
			if drop_below_flag:
				continue

			# Now calculate post_fair
			post_min_fair=1.0
			post_comm_len=sum(post_colors)
			for col_ind,_col in enumerate(color_list):
				if post_colors[col_ind]==0:
					post_min_fair=0.0 
				else:
					post_col_score=0.0
					if post_comm_len>post_colors[col_ind]:
						post_col_score=post_colors[col_ind]/(post_comm_len-post_colors[col_ind])
					# Set to min if less than current
					if post_col_score<post_min_fair:
						post_min_fair=post_col_score
			# Weigh by community size ratio for post_fair
			post_fair=(K_cols-1)*post_comm_len*post_min_fair/n


			#print(f"Moving node {u}: curr_fair={curr_fair}, post_fair={post_fair}")


			# For each neighbour of u: 
			for nbr_comm, wt in w2c.items():

				# Calculate new modularity
				mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)

				# Calculate gain for move to nbr_comm
				new_curr_len=sum([partition_colors[nbr_comm][col] for col_ind,col in enumerate(color_list)])
				# Weigh by community size ratio. Assumes already normalized score
				new_score = partition_colors[nbr_comm]["score"]

				new_colors = [partition_colors[nbr_comm][col]+curr_colors[col_ind] for col_ind,col in enumerate(color_list)]
				# Calculate fairness score for newly created community (plus u)
				new_min_fair=1.0
				new_comm_len=sum(new_colors)
				for col_ind,_col in enumerate(color_list):
					if new_colors[col_ind]==0:
						new_min_fair=0.0 
					else:
						new_col_score=0.0
						if new_comm_len>new_colors[col_ind]:
							new_col_score=new_colors[col_ind]/(new_comm_len-new_colors[col_ind])
						# Set to min if less than current
						if new_col_score<new_min_fair:
							new_min_fair=new_col_score
				# Weigh by community length ratio for new_fair. Times Kcols
				new_fair=(K_cols-1) * new_min_fair * new_comm_len / n

				# Overall fairness score gain
				# new_fair (nbr) + post_fair (comms(u)) - new_score (nbr) - curr_fair (comms(u))
				fair_gain = (new_fair - new_score) + (post_fair - curr_fair)

				# Calculate new opt score.
				opt_gain = alpha * mod_gain + (1-alpha) * fair_gain

				# If opt gain tops previous gain found, set as new best
				if opt_gain > opt_best and (opt_gain - opt_best) > threshold:
					# # @DEBUG
					# print(f"Better found move: {u}->comm#{nbr_comm}")
					# print(f"u curr_f={curr_fair}, u post_f={post_fair}")
					# print(f"v curr_f={new_score}, v post_f={new_fair}")
					# print(f"fair gain={fair_gain}, mod_gain={mod_gain}")
					# print(f"opt gain={opt_gain}, (opt_best={opt_best})")
					# print("-----------")

					opt_best=opt_gain
					comm_best=nbr_comm

					new_fair_update=new_fair
					post_fair_update=post_fair


			# Then finalize move if necessary
			Stot[comm_best]+=degree
			if comm_best != comms[u]:
				# Finalize move

				# Get nodes in com
				com = G.nodes[u].get("nodes", {u})


				# Update partition, remove com nodes from comms(u)
				partition[comms[u]].difference_update(com)
				# Update inner partition, remove u from comms(u)
				inner_partition[comms[u]].remove(u)
				# Update partition, add com nodes to comm_best
				partition[comm_best].update(com)
				# Update inner partition, add u to comm_best
				inner_partition[comm_best].add(u)
				# Signal improvement
				improvement = True
					
				# Comment for infinite loop on a=0
				#n_moves += 1


				# Update new community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comm_best][col]+=curr_colors[col_ind]
				partition_colors[comm_best]["score"]=new_fair_update
				
				# Update old community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comms[u]][col]-=curr_colors[col_ind]
				partition_colors[comms[u]]["score"]=post_fair_update


				# Change new best community for u
				comms[u] = comm_best



	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))	
	partition_colors_new = list(filter(lambda ls: _full_partition_colors(ls,color_list), partition_colors))


	return partition, inner_partition, improvement, partition_colors_new



# Calculate one level of partitions based on full Obj, with F_exp penalty
@py_random_state("seed")
def _calculate_partition_fexp(G, n, m, partition, colors, color_dist, partition_cols, phi, alpha=0.9, resolution=1, threshold=0.0000001, is_directed=False, seed=None):
	# At start, assign each node to a community 
	comms = {u: i for i,u in enumerate(G.nodes())}
	inner_partition = [{u} for u in G.nodes()]

	color_list=color_dist.keys()
	K_cols=len(color_list)

	partition_colors=partition_cols.copy()
	original_partition=partition.copy()

	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	# Get all neighbours, including R/B
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
	
	# Get sum of edge weights for all B/R nodes
	sum_all = {u: len(nbrs[u]) for u in G}

	degrees=dict(G.degree(weight="weight"))
	Stot=list(degrees.values())

	# Do random shuffle on seed
	rand_nodes=list(G.nodes)
	seed.shuffle(rand_nodes)

	# Start calculating movements to other communities
	n_moves=1
	improvement=False


	while n_moves>0:
		n_moves=0
			
		# For each node:
		for u in rand_nodes:
			
			## Start with opt score for its current community?
			#  Idea: the minimum gain score is 0 for modularity and the previous fair
			#     score for the current community =>a*0+(1-a)*curr score
			opt_best=0.0
			#opt_best=(1-alpha)*partition_colors[comms[u]]["score"]
			comm_best=comms[u]
			new_fair_update=0.0
			post_fair_update=0.0

			# Get degrees
			degree=degrees[u]
			Stot[comm_best]-=degree

			# For each node, calculate the weights of its neighbours in the same comm
			w2c=neighbor_weights(nbrs[u],comms)

			
			# Calculate modularity remove cost
			mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)

			# Calculate fairness (balance) current cost
			#### @TODO: slowest part, probably. Can be optimized?
			com = G.nodes[u].get("nodes", {u})


			# For nodes under u: calculate their color scores
			curr_colors=[0 for _col in color_list]
			for i in com:
				for col_ind, col in enumerate(color_list):
					if colors[i]==col:
						curr_colors[col_ind]+=1

			# Curr fair: all nodes in comms(u)
			all_colors=[partition_colors[comms[u]][col] for col in color_list]			
	
			min_fair=1.0
			sum_pnodes=0
			for col_ind, col in enumerate(color_list):
				sum_color=all_colors[col_ind]
				sum_pnodes+=sum_color
				# Min=0 if any color is 0
				if sum_color==0: 
					min_fair=0.0
				else:
					up_score=0.0
					if len(com)>sum_color:
						# See if minimum color score is less than current min
						up_score=sum_color/(len(com)-sum_color)

					# Set to min if less than current
					if up_score<min_fair:
						min_fair=up_score

			# Current fairness score for u
			curr_fair=(K_cols-1)*min_fair


			# For current partition: calculate current F_exp
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*len(com)/n)
			curr_n_extra=len(com)-sum_dist

			# Calculate current F_exp(c_i)
			curr_f_exp=(K_cols*phi*len(com)+(phi+K_cols-1-(phi*K_cols))*curr_n_extra) /((K_cols*len(com) + (phi-1)*curr_n_extra))


			# Calculate score for current c_i minus the penalty
			if len(com)>=K_cols:
				curr_fscore=min(1.0,1-(curr_f_exp-((K_cols-1)*curr_fair)))
			else:
				curr_fscore=0.0



			# Calculate post_fair score of current community after moving u
			post_colors=[all_colors[col_ind]-curr_colors[col_ind] for col_ind,_col in enumerate(color_list)]
			
			# Important: add failsafe here. If post drops <0, do not allow move. cont
			drop_below_flag=False
			for col_ind,_col in enumerate(color_list):
				if post_colors[col_ind]<0: 
					drop_below_flag=True
					break
			if drop_below_flag:
				continue

			# Now calculate post_fair
			post_min_fair=1.0
			post_comm_len=sum(post_colors)
			for col_ind,_col in enumerate(color_list):
				if post_colors[col_ind]==0:
					post_min_fair=0.0 
				else:
					post_col_score=0.0
					if post_comm_len>post_colors[col_ind]:
						post_col_score=post_colors[col_ind]/(post_comm_len-post_colors[col_ind])
					# Set to min if less than current
					if post_col_score<post_min_fair:
						post_min_fair=post_col_score

			# Weigh by community length for post_fair
			post_fair=(K_cols-1)*post_min_fair


			# For post partition: calculate F_exp
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*post_comm_len/n)
			post_n_extra=post_comm_len-sum_dist

			# Calculate current F_exp(c_i)
			if post_comm_len==0:
				post_f_exp=0.0
			else:
				post_f_exp=(K_cols*phi*post_comm_len+(phi+K_cols-1-(phi*K_cols))*post_n_extra) /((K_cols*post_comm_len + (phi-1)*post_n_extra))

			# Calculate score for current c_i minus the penalty
			if post_comm_len>=K_cols:
				post_fscore=min(1.0,1-(post_f_exp-((K_cols-1)*post_fair)))
			else:
				post_fscore=0.0


			# Calculate current community loss from move
			loss_fscore=((post_fscore*post_comm_len)-(curr_fscore*len(com)))/n

			# # @DEBUG
			# print(f"Moving node #{u}. Curr_fair={curr_fair}, fscore={curr_fscore}, post_fair={post_fair}, post_fscore={post_fscore}, loss={loss_fscore}")


			# For each neighbour of u: 
			for nbr_comm, wt in w2c.items():

				# Calculate new modularity
				mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)

				# Calculate gain for move to nbr_comm
				new_curr_len=sum([partition_colors[nbr_comm][col] for col_ind,col in enumerate(color_list)])
				new_score = partition_colors[nbr_comm]["score"]
				## Assuming here: old score already has penalty on top


				new_colors = [partition_colors[nbr_comm][col]+curr_colors[col_ind] for col_ind,col in enumerate(color_list)]
				# Calculate fairness score for newly created community (plus u)
				new_min_fair=1.0
				new_comm_len=sum(new_colors)
				for col_ind,_col in enumerate(color_list):
					if new_colors[col_ind]==0:
						new_min_fair=0.0 
					else:
						new_col_score=0.0
						if new_comm_len>new_colors[col_ind]:
							new_col_score=new_colors[col_ind]/(new_comm_len-new_colors[col_ind])
						# Set to min if less than current
						if new_col_score<new_min_fair:
							new_min_fair=new_col_score
				new_fair=new_min_fair


				# Also calculate new score of v with penalty
				# For post partition: calculate F_exp
				# First, calculate extra nodes for the community
				sum_dist=0
				for col in color_list:
					sum_dist+=math.floor(color_dist[col]*new_comm_len/n)
				new_n_extra=new_comm_len-sum_dist

				# Calculate new F_exp(c_i)
				if new_comm_len==0:
					new_f_exp=0.0
				else:
					new_f_exp=(K_cols*phi*new_comm_len+(phi+K_cols-1-(phi*K_cols))*new_n_extra) /((K_cols*new_comm_len + (phi-1)*new_n_extra))

				# Calculate score for current c_i minus the penalty
				if new_comm_len>=K_cols:
					new_fscore=min(1.0,1-(new_f_exp-((K_cols-1)*new_fair)))
				else:
					new_fscore=0.0

				# Calculate gain on v:
				gain_fscore=((new_fscore*new_comm_len)-(new_score*new_curr_len))/n


				
				### ------ UPDATE GAINS HERE ------
				# New gain including penalties: gain_v + loss_u
				# Overall fairness score gain with penalties (normalize weighted by n)
				fair_gain = (gain_fscore + loss_fscore)

				# Calculate new opt score
				opt_gain = alpha * mod_gain + (1-alpha) * fair_gain


				# # @DEBUG
				# print(f"Test ({u}->{nbr_comm}) opt_gain={opt_gain}, mod_gain={mod_gain}, fair_gain={fair_gain}")

				# If opt gain tops previous gain found, set as new best
				if opt_gain > opt_best and (opt_gain - opt_best) > threshold:


					# # @DEBUG
					# print(f"Better found move: {u}->comm#{nbr_comm}")
					# print(f"u curr_f={curr_fscore}, u post_f={post_fscore}")
					# print(f"v curr_f={new_score}, v post_f={new_fscore}")
					# print(f"fair gain={fair_gain}, mod_gain={mod_gain}")
					# print(f"opt gain={opt_gain}, (opt_best={opt_best})")
					# print("-----------")

					opt_best=opt_gain
					comm_best=nbr_comm

					new_fair_update=new_fscore
					post_fair_update=post_fscore


			# Then finalize move if necessary
			Stot[comm_best]+=degree
			if comm_best != comms[u]:
				# Finalize move

				# Get nodes in com
				com = G.nodes[u].get("nodes", {u})

				# Update partition, remove com nodes from comms(u)
				partition[comms[u]].difference_update(com)
				# Update inner partition, remove u from comms(u)
				inner_partition[comms[u]].remove(u)
				# Update partition, add com nodes to comm_best
				partition[comm_best].update(com)
				# Update inner partition, add u to comm_best
				inner_partition[comm_best].add(u)
				# Signal improvement
				improvement = True
					
				# Comment for infinite loop on a=0
				#n_moves += 1


				# Update new community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comm_best][col]+=curr_colors[col_ind]
				partition_colors[comm_best]["score"]=new_fair_update
				
				# Update old community colors
				for col_ind,col in enumerate(color_list):
					partition_colors[comms[u]][col]-=curr_colors[col_ind]
				partition_colors[comms[u]]["score"]=post_fair_update

				# Change new best community for u
				comms[u] = comm_best


	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))
	partition_colors_new = list(filter(lambda ls: _full_partition_colors(ls,color_list), partition_colors))
		

	return partition, inner_partition, improvement, partition_colors_new

# Calculate one level of partitions on full Obj=a*deltaQ+(1-a)*deltaF; 
# where deltaQ is modularity gain and deltaF is modularity fairness gain
@py_random_state("seed")
def _calculate_partition_fmody(G, n, m, partition, colors, color_dist, phi, alpha=0.9, resolution=1, threshold=0.000001, is_directed=False, seed=None):
    # At start, assign each node to a community 
    comms = {u: i for i,u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]

    color_list=color_dist.keys()

    original_partition=partition.copy()

    # @TODO: fix when supporting DiGraphs
    if is_directed==True:
        print("Directed networks not supported.")
        return None

    # Get all neighbours, including R/B
    nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}

    # Get sum of edge weights for all B/R nodes
    sum_all = {u: len(nbrs[u]) for u in G}

    degrees=dict(G.degree(weight="weight"))
    Stot=list(degrees.values())

    # Do random shuffle on seed
    rand_nodes=list(G.nodes)
    seed.shuffle(rand_nodes)
    
    # Precompute total degree for each community, degree for each color in a community
    community_degree = defaultdict(float)
    community_degrees_red = defaultdict(float)
    community_degrees_blue = defaultdict(float)
    
    # Precompute intra-community edge counts
    community_red_edges = defaultdict(float)
    community_blue_edges = defaultdict(float)
    
    # Compute initial degree sum for each community and initial intra-community edge counts
    for u in G.nodes():
        comm = comms[u]
        deg = G.degree(u, weight="weight")
        community_degree[comm] += G.degree(u, weight="weight")
        # Color specific degree sums
        if G.nodes[u]["color"] == "red":
            community_degrees_red[comm] += deg
        elif G.nodes[u]["color"] == "blue":
            community_degrees_blue[comm] += deg
        
    for u, v, data in G.edges(data=True):
        comm = comms[u]
        if comm == comms[v]:
            w = data.get("weight", 1.0)
            if G.nodes[u]["color"] == "red" or G.nodes[v]["color"] == "red":
                community_red_edges[comm] += w
            if G.nodes[u]["color"] == "blue" or G.nodes[v]["color"] == "blue":
                community_blue_edges[comm] += w

    # Start calculating movements to other communities
    n_moves=1
    improvement=False       

    while n_moves>0:
        n_moves=0
            
        # For each node:
        for u in rand_nodes:
            # Start with opt score for its current community?
            #  Idea: the minimum gain score is 0 for modularity and the previous fair
            #     score for the current community =>a*0+(1-a)*curr score
            
            # Initialize the best optimized value as 0 and best community as the initial community
            opt_best=0.0
            comm_best=comms[u]

            # Get degrees
            degree=degrees[u]
            Stot[comm_best]-=degree

            # For each node, calculate the weights of its neighbours in the same comm
            w2c=neighbor_weights(nbrs[u],comms)
            # print("w2c = ", w2c)
            
            # Calculate modularity remove cost
            mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)
            # print("mod_rem = ", mod_rem)

            # Calculate modularity fairness current cost
            #### @TODO: slowest part, probably. Can be optimized?
            com = G.nodes[u].get("nodes", {u})
        
            # Current modularity fairness for existing community
            # curr_fscore = modularity_fairness(G, [com], color_dist, colors)
            # print(f"Moving node {u}: curr_fscore={curr_fscore}")

            # For each neighbour of u: 
            for nbr_comm, wt in w2c.items():

                # Calculate new modularity
                mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)
                # print("Modularity gain = ", mod_gain)

                # Calculate fairness gain
                fair_gain = modularity_fairness_gain(G, m, comms, u, nbr_comm, community_degree, community_degrees_red, community_degrees_blue, community_red_edges, community_blue_edges)
                # print("Modularity Fairness gain = ", fair_gain)

                # Calculate new opt score.
                opt_gain = alpha * mod_gain + (1-alpha) * fair_gain
                # print("Opt gain = ", opt_gain)

                # If opt gain tops previous gain found, set as new best
                if opt_gain > opt_best and (opt_gain - opt_best) > threshold:
                    # # @DEBUG
                    # print(f"Better found move: {u}->comm#{nbr_comm}")
                    # print(f"u curr_f={curr_fair}, u post_f={post_fair}")
                    # print(f"v curr_f={new_score}, v post_f={new_fair}")
                    # print(f"fair gain={fair_gain}, mod_gain={mod_gain}")
                    # print(f"opt gain={opt_gain}, (opt_best={opt_best})")
                    # print("-----------")

                    opt_best=opt_gain
                    comm_best=nbr_comm


            # Then finalize move if necessary
            Stot[comm_best]+=degree
            if comm_best != comms[u]:
                # Finalize move

                # Get nodes in com
                com = G.nodes[u].get("nodes", {u})
                # Update partition, remove com nodes from comms(u)
                partition[comms[u]].difference_update(com)
                # Update inner partition, remove u from comms(u)
                inner_partition[comms[u]].remove(u)
                # Update partition, add com nodes to comm_best
                partition[comm_best].update(com)
                # Update inner partition, add u to comm_best
                inner_partition[comm_best].add(u)
                # Signal improvement
                improvement = True
                    
                # Comment for infinite loop on a=0
                # n_moves += 1

                # Change new best community for u
                comms[u] = comm_best

    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))

    return partition, inner_partition, improvement

# Calculate one level of partitions on full Obj=a*deltaQ+(1-a)*deltaF; 
# where deltaQ is modularity gain and deltaF is diversity fairness gain
@py_random_state("seed")
def _calculate_partition_diversity(G, n, m, partition, colors, color_dist, phi, alpha=0.9, resolution=1, threshold=0.000001, is_directed=False, seed=None):
    # At start, assign each node to a community 
    comms = {u: i for i,u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]

    color_list=color_dist.keys()

    original_partition=partition.copy()

    # @TODO: fix when supporting DiGraphs
    if is_directed==True:
        print("Directed networks not supported.")
        return None

    # Get all neighbours, including R/B
    nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}

    # Get sum of edge weights for all B/R nodes
    sum_all = {u: len(nbrs[u]) for u in G}

    degrees=dict(G.degree(weight="weight"))
    Stot=list(degrees.values())

    # Do random shuffle on seed
    rand_nodes=list(G.nodes)
    seed.shuffle(rand_nodes)
    
    # Precompute intra-community edge counts where a node is red and other is blue in a community
    community_edges_rb = defaultdict(float)
    
    # Precompute degree for each color in a community
    community_red_degress = defaultdict(float)
    community_blue_degress = defaultdict(float)
    
    # Compute degree sum for each color within communities
    for u in G.nodes():
        comm = comms[u]
        deg = G.degree(u, weight="weight")
        # Color specific degree sums
        if colors[u] == "red":
            community_red_degress[comm] += deg
        elif colors[u] == "blue":
            community_blue_degress[comm] += deg
        
    for u, v, data in G.edges(data=True):
        comm = comms[u]
        if comm == comms[v]:
            w = data.get("weight", 1.0)
            if (colors[u] == "red" and colors[v] == "blue") or (colors[u] == "blue" and colors[v] == "red"):
                community_edges_rb[comm] += w

    # Start calculating movements to other communities
    n_moves=1
    improvement=False       

    while n_moves>0:
        n_moves=0
            
        # For each node:
        for u in rand_nodes:
            # Start with opt score for its current community?
            #  Idea: the minimum gain score is 0 for modularity and the previous fair
            #     score for the current community =>a*0+(1-a)*curr score
            
            # Initialize the best optimized value as 0 and best community as the initial community
            opt_best=0.0
            comm_best=comms[u]

            # Get degrees
            degree=degrees[u]
            Stot[comm_best]-=degree

            # For each node, calculate the weights of its neighbours in the same comm
            w2c=neighbor_weights(nbrs[u],comms)
            # print("w2c = ", w2c)
            
            # Calculate modularity remove cost
            mod_rem = -w2c[comm_best]/m + resolution*(Stot[comm_best]*degree)/(2* m**2)
            # print("mod_rem = ", mod_rem)

            # Calculate modularity fairness current cost
            #### @TODO: slowest part, probably. Can be optimized?
            com = G.nodes[u].get("nodes", {u})
        
            # Current modularity fairness for existing community
            # curr_fscore = modularity_fairness(G, [com], color_dist, colors)
            # print(f"Moving node {u}: curr_fscore={curr_fscore}")

            # For each neighbour of u: 
            for nbr_comm, wt in w2c.items():

                # Calculate new modularity
                mod_gain = mod_rem + wt/m - resolution*(Stot[nbr_comm]*degree)/(2* m**2)
                # print("Modularity gain = ", mod_gain)

                # Calculate fairness gain
                fair_gain = diversity_fairness_gain(G, m, comms, u, nbr_comm, community_edges_rb, community_red_degress, community_blue_degress, colors)
                # print("Modularity Fairness gain = ", fair_gain)

                # Calculate new opt score.
                opt_gain = alpha * mod_gain + (1-alpha) * fair_gain
                # print("Opt gain = ", opt_gain)

                # If opt gain tops previous gain found, set as new best
                if opt_gain > opt_best and (opt_gain - opt_best) > threshold:
                    # # @DEBUG
                    # print(f"Better found move: {u}->comm#{nbr_comm}")
                    # print(f"u curr_f={curr_fair}, u post_f={post_fair}")
                    # print(f"v curr_f={new_score}, v post_f={new_fair}")
                    # print(f"fair gain={fair_gain}, mod_gain={mod_gain}")
                    # print(f"opt gain={opt_gain}, (opt_best={opt_best})")
                    # print("-----------")

                    opt_best=opt_gain
                    comm_best=nbr_comm


            # Then finalize move if necessary
            Stot[comm_best]+=degree
            if comm_best != comms[u]:
                # Finalize move

                # Get nodes in com
                com = G.nodes[u].get("nodes", {u})
                # Update partition, remove com nodes from comms(u)
                partition[comms[u]].difference_update(com)
                # Update inner partition, remove u from comms(u)
                inner_partition[comms[u]].remove(u)
                # Update partition, add com nodes to comm_best
                partition[comm_best].update(com)
                # Update inner partition, add u to comm_best
                inner_partition[comm_best].add(u)
                # Signal improvement
                improvement = True
                    
                # Comment for infinite loop on a=0
                # n_moves += 1

                # Change new best community for u
                comms[u] = comm_best

    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))

    return partition, inner_partition, improvement

@py_random_state("seed")
def _calculate_partition_diversity_paper(G, n, m, iterationNum, partition, diversity_dist, nodesList, colors, phi, alpha=0.9, resolution=1, threshold=0.000001, is_directed=False, seed=None):
	node2com = {u: i for i,u in enumerate(G.nodes())}

	inner_partition = [{u} for u in G.nodes()]
 
	# @TODO: fix when supporting DiGraphs
	if is_directed==True:
		print("Directed networks not supported.")
		return None

	degrees = dict(G.degree(weight="weight"))
	Stot = list(degrees.values())
	nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}

	degrees_r = dict(G.nodes(data="red_weight"))
	Stot_r = list(degrees_r.values())
	nbrs_r = {u: {v: data["r_weight"] for v, data in G[u].items() if v != u} for u in G}

	degrees_b = dict(G.nodes(data="blue_weight"))
	Stot_b = list(degrees_b.values())
	nbrs_b = {u: {v: data["b_weight"] for v, data in G[u].items() if v != u} for u in G}
 
	nbrs_inter = {u: {v: data["inter_weight"] for v, data in G[u].items() if v != u} for u in G}
 
	randNodes = nodesList
	nb_moves = 1
	improvement = False
	iteNum = 0
	timesInWhile = 0
	checkParameter = int(len(G.nodes())/2)
 
	while(nb_moves > 0):
		if timesInWhile>checkParameter:
			break
		timesInWhile += 1
		iteNum += 1
		nb_moves = 0
		changes_foundNum = 0
		for u in randNodes:
			best_fair_gain = 0
			best_fair_remove_cost = 0
			best_gain = 0
			best_diversity = 0
			opt_best=0.0

			best_com = node2com[u]
			weights2com = neighbor_weights(nbrs[u], node2com)
			weights2comR = neighbor_weights(nbrs_r[u], node2com)
			weights2comB = neighbor_weights(nbrs_b[u], node2com)
			weights2comInter = neighbor_weights(nbrs_inter[u], node2com)
   
			degree = degrees[u]
			Stot[best_com] -= degree
			remove_cost = -weights2com[best_com] / m + resolution * (
                    Stot[best_com] * degree
                ) / (2 * m**2)

			degree_r = degrees_r[u]
			Stot_r[best_com] -= degree_r
			
			if iterationNum>0:
				remove_cost_R = -weights2comR[best_com] / m + resolution * (
					Stot_r[best_com] * degree_r) / (2 * m**2)
			else:
				remove_cost_R = degree_r / (2 * m**2)
    
			degree_b = degrees_b[u]   
			Stot_b[best_com] -= degree_b
   
			if iterationNum>0:
				remove_cost_B = -weights2comB[best_com] / m + resolution * (
					Stot_b[best_com] * degree_b) / (2 * m**2)
			else:
				remove_cost_B = degree_b / (2 * m**2)

			diversityModularity_before_best_com = diversity_dist[best_com]
			remove_cost_inter = -weights2comInter[best_com] / m + resolution * ((Stot_r[best_com] * degree_b)+(Stot_b[best_com] * degree_r)) / (2 * m**2)

			diversityModularity_after_best_com = diversityModularity_before_best_com + remove_cost_inter
   
			for nbr_com, wt in weights2com.items():
				wtR = weights2comR[nbr_com]
				wtB = weights2comB[nbr_com]
				wtInter = weights2comInter[nbr_com]
				gain = (
					remove_cost
					+ wt / m
					- resolution * (Stot[nbr_com] * degree) / (2 * m**2)
				)
                    
				gain = (
					wt / m
					- resolution * (Stot[nbr_com] * degree) / (2 * m**2)
				)

				gain_r = (
					wtR / m
					- resolution * (Stot_r[nbr_com] * degree_r) / (2 * m**2)
				)
				
				gain_b = (
					wtB / m
					- resolution * (Stot_b[nbr_com] * degree_b) / (2 * m**2)
				)
				
				gain_rb = (wtInter / m) - resolution * ((Stot_r[nbr_com] * degree_b)+(Stot_b[nbr_com] * degree_r)) / (2 * m**2)

				diversityModularity_before_nbr_com = diversity_dist[nbr_com]
				diversityModularity_after_nbr_com = diversityModularity_before_nbr_com + gain_rb
				
				fair_Inter_before = abs(diversityModularity_before_nbr_com +diversityModularity_before_best_com)
				fair_Inter_after = abs(diversityModularity_after_nbr_com +diversityModularity_after_best_com)
				
				mod_gain = gain+remove_cost
				diversity_gain = gain_rb+remove_cost_inter
				# check_inter = abs(fair_Inter_after) - abs(fair_Inter_before)
				# check_in = best_diversity<=diversity_gain and mod_gain>best_gain
				opt_new = alpha * mod_gain + (1-alpha) * diversity_gain	
					
				if (opt_new - opt_best) > threshold and nbr_com != node2com[u]: # check_in and check_inter>0 and 
					changes_foundNum+=1
					best_com = nbr_com
					best_diversity = diversity_gain
					best_gain = mod_gain
					opt_best = opt_new

			check = True
			if check:
				Stot[best_com] += degree
				Stot_r[best_com] += degree_r
				Stot_b[best_com] += degree_b

				if best_com != node2com[u] and best_gain>=0:
					connectedBool = True
					if connectedBool:
						com = G.nodes[u].get("nodes", {u})
						
						partition[node2com[u]].difference_update(com)
						inner_partition[node2com[u]].remove(u)
						partition[best_com].update(com)
						inner_partition[best_com].add(u)
						improvement = True
						nb_moves += 1
						node2com[u] = best_com

	partition = list(filter(len, partition))
	inner_partition = list(filter(len, inner_partition))
	return partition, inner_partition, improvement
