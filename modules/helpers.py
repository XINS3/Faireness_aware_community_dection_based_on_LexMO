import math
from collections import defaultdict

import networkx as nx

# Calculates average balance fairness of G for a given partition into communities
def fairness_base(G, partition, color_dist):
	colors=nx.get_node_attributes(G, "color")
	n=G.number_of_nodes()

	color_list=color_dist.keys()
	K_cols=len(color_list)

	# If n==0: return 0. Should not happen, but good failsafe
	if n==0: return 0

	sum_scores=0.0
	F_dist=[]

	# If color list length==1: balance is 1 by default
	if len(color_list)<=1:
		return [1.0 for _c in partition]

	# For all communities discovered
	for i, ci in enumerate(partition):
		# If community is populated:
		n_ci=len(ci)
		if n_ci>0:
			# For all nodes u in ci, check sums of colors
			sum_cols=[0 for _c in color_list]
			for u in ci:
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

			# Set min_balance as the score. Normalize by K_cols s.t. max score is 1
			balance_ci=(K_cols-1) * min_balance * n_ci / n

			# Add to total sum (weighted by n_ci)
			sum_scores+=balance_ci
			F_dist.append(balance_ci)

	return sum_scores, F_dist



# Calculates average balance fairness for G with F_exp penalty
def fairness_fexp(G, partition, color_dist):
	colors=nx.get_node_attributes(G, "color")
	n=G.number_of_nodes()

	color_list=color_dist.keys()
	K_cols=len(color_list)

	# Calculate phi
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If n==0: return 0. Should not happen, but good failsafe
	if n==0: return 0

	sum_scores=0.0
	F_dist=[]

	# If color list length==1: balance is 1 by default
	if len(color_list)<=1:
		return [1.0 for _c in partition]

	# For all communities discovered
	for i, ci in enumerate(partition):
		# If community is populated:
		n_ci=len(ci)
		if n_ci>0:
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*n_ci/n)
			n_extra=n_ci-sum_dist

			# Calculate F_exp(c_i)
			f_exp=(K_cols*phi*n_ci+(phi+K_cols-1-(phi*K_cols))*n_extra) /((K_cols*n_ci + (phi-1)*n_extra))

			## Also calculate F(c_i)
			# For all nodes u in ci, check sums of colors
			sum_cols=[0 for _c in color_list]
			for u in ci:
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

			# Set min_balance as the score. Normalize by K-cols st max score =1
			balance_ci=(K_cols-1)*min_balance

			# Get final score
			if n_ci>=K_cols:
				fscore_ci=min(1.0,1-(f_exp-balance_ci))
			else:
				fscore_ci=0.0


			# Add to total sum (weighted by n_ci)
			ci_score=(n_ci*fscore_ci)
			sum_scores+=ci_score
			F_dist.append(ci_score/n)

	return sum_scores/n, F_dist


# Helper function to remove all empty dicts
def _full_partition_colors(part_dict,color_list):
	# Iterate over all colors. If any of their sums is non-zero, return True
	for c in color_list:
		if part_dict[c]!=0: return True
	# Otherwise, if partition is empty: return False to discard.
	return False


# Get weights between node and its neighbor communities. Also for blue and red nodes
def neighbor_weights(nbrs, comms):
	weights=defaultdict(float)
	for nbr, w in nbrs.items():
		c = comms[nbr]
		weights[c]+=w
	return weights
	

# Generate a new graph based on the partitions of a given graph.
# Also update partition colors
def _gen_graph(G, partition, colors, diversity_flag=False):
	H = G.__class__()
	node2com = {}
	partition_colors = {}
	for i, part in enumerate(partition):
		nodes = set()
		if diversity_flag:
			red_degree = 0
			blue_degree = 0
			inter_degree = 0
		for node in part:
			partition_colors[i]={}
			node2com[node] = i
			if diversity_flag:
				red_degree +=G.nodes[node]["red_weight"]
				blue_degree +=G.nodes[node]["blue_weight"]
				inter_degree += G.nodes[node]["inter_weight"]
			nodes.update(G.nodes[node].get("nodes", {node}))

			color = colors.get(node)
			if color:
				partition_colors[i][color] = partition_colors[i].get(color, 0) + 1

		H.add_node(i, nodes=nodes, color=partition_colors[i], red_weight= 0, blue_weight = 0, inter_weight= 0)
  
		if diversity_flag:
			temp_red = H.nodes[i]["red_weight"]
			temp_blue = H.nodes[i]["blue_weight"]
			temp_inter = H.nodes[i]["inter_weight"]

			H.add_node(i, nodes=nodes, color=partition_colors[i], red_weight=red_degree+temp_red, blue_weight=blue_degree+temp_blue, inter_weight=inter_degree+temp_inter)

	for node1, node2, wts in G.edges(data=True):
		wt = wts["weight"]
		com1 = node2com[node1]
		com2 = node2com[node2]
		temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
		if diversity_flag:
			wt_red = wts["r_weight"]
			wt_blue = wts["b_weight"]
			wt_inter = wts["inter_weight"]
			temp_red = H.get_edge_data(com1, com2, {"r_weight": 0})["r_weight"]
			temp_blue = H.get_edge_data(com1, com2, {"b_weight": 0})["b_weight"]
			temp_inter = H.get_edge_data(com1, com2, {"inter_weight":0})["inter_weight"]
			H.add_edge(com1, com2, weight=wt + temp, r_weight=wt_red+temp_red, b_weight=wt_blue+temp_blue, inter_weight=wt_inter+temp_inter)
		else:
			H.add_edge(com1, com2, weight=wt + temp)
	return H, partition_colors

# Convert a Multigraph to normal Graph
def _convert_multigraph(G, weight, is_directed):
    if is_directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H

# Calculate modularity fairness for a particular partition
def modularity_fairness(G, partition, color_dist, colors):
    n = G.number_of_nodes()
    m = G.size(weight="weight")

    color_list=color_dist.keys()
    
    # If n==0: return 0. Should not happen, but good failsafe
    if n==0: return 0

    sum_scores = 0.0
    F_dist = []
    # partition_summary = []
    
    # print("Partition in modularity_fairness function = ", partition)

    # Calculate modularity fairness for each community
    for i, ci in enumerate(partition):
        if not isinstance(ci, (set, list, tuple, int)):
            raise TypeError(f"Unexpected type in partition: {type(ci)}, value: {ci}")
        nodes_in_ci = [node for node in ci] # list(ci) if isinstance(ci, (set, list, tuple)) else [ci] # [node for node in ci]
        # print(f"Processing community: {nodes_in_ci} (Type: {type(nodes_in_ci)})")
        n_ci = len(nodes_in_ci)
        # print("Type of nodes_in_ci = ", type(nodes_in_ci[0]))
        # print(f"Processing community: {ci} (Type: {type(ci)})")
        if not all(isinstance(node, int) for node in nodes_in_ci):
            raise ValueError(f"Non-integer node found: {nodes_in_ci}")
        if n_ci > 0:
            # Calculate number of intra-community edges and sum of degrees for the community
            try:
                L_c = sum(1 for u, v in G.edges(nodes_in_ci))
            except Exception as e:
                print(f"Error in edge calculation: {e}")
                continue
            # print("Print G.degree of nodes = ", [G.degree(node) for node in nodes_in_ci])
            try:
                temp = [G.degree(node) for node in nodes_in_ci]
                d_c = sum(temp)
            except Exception as e:
                print(f"Error in degree calculation: {e}")
                continue

            # Modularity fairness for community Ci
            Q_Ci = L_c/m - (d_c * d_c)/(4 * m * m)

            # Calculate number of intra-community edges with atleast one and sum of degrees for red  
            L_c_R = sum(1 for u, v in G.edges(nodes_in_ci) if ("red" in (colors[u], colors[v])))
            d_c_R = sum(G.degree(u) for u in nodes_in_ci if colors[u] == "red")
            
            # Calculate number of intra-community edges with atleast one and sum of degrees for blue  
            L_c_B = sum(1 for u, v in G.edges(nodes_in_ci) if ("blue" in (colors[u], colors[v])))
            d_c_B = sum(G.degree(u) for u in nodes_in_ci if colors[u] == "blue")

            # Calculate modularity fairness for each color in the community
            Q_Ci_R = L_c_R/m - (d_c * d_c_R)/(4 * m * m)
            Q_Ci_B = L_c_B/m - (d_c * d_c_B)/(4 * m * m)

            # Modularity fairness for community Ci
            if Q_Ci == 0:
                mod_fair_Ci = 0
            else:
                mod_fair_Ci = (Q_Ci_R - Q_Ci_B) / abs(Q_Ci)
            weighted_mod_fair_Ci = mod_fair_Ci * n_ci/n

            sum_scores += weighted_mod_fair_Ci
            F_dist.append(weighted_mod_fair_Ci)

    return sum_scores, F_dist

# Helper function to calculate modularity fairness gain when a node is moved to a new community
def modularity_fairness_gain(G, m, communities, node, new_community, community_degree, community_degrees_red, community_degrees_blue, community_red_edges, community_blue_edges):
	node_color = G.nodes[node]["color"] 	# red and blue, this is the only case defined in this algorithm
 
	curr_community = communities[node]

	# Sum of degrees of nodes in old and new communities
	sum_weights_C_old = community_degree[curr_community]
	sum_weights_C_new = community_degree[new_community]
 
	# Sum of degress of red and blue nodes in each community
	sum_tot_R_old = community_degrees_red[curr_community]
	sum_tot_B_old = community_degrees_blue[curr_community]

	sum_tot_R_new = community_degrees_red[new_community]
	sum_tot_B_new = community_degrees_blue[new_community]

	# Count intra-community edges for red and blue nodes
	L_C_R_old = community_red_edges[curr_community]
	L_C_B_old = community_blue_edges[curr_community]

	L_C_R_new = community_red_edges[new_community]
	L_C_B_new = community_blue_edges[new_community]

	# Calculate old modularity fairness
	Q_F_R_old = (L_C_R_old/m) - ((sum_weights_C_old * sum_tot_R_old)/(4 * m * m))
	Q_F_B_old = (L_C_B_old/m) - ((sum_weights_C_old * sum_tot_B_old)/(4 * m * m))
	Q_F_old = Q_F_R_old - Q_F_B_old
 
	node_degree = G.degree(node)
	# Sum of intra-community edge count for new and current communities
	sum_intra_comm_new = sum(1 for neighbor in G.neighbors(node) if communities[neighbor] == new_community)
	sum_intra_comm_curr = sum(1 for neighbor in G.neighbors(node) if communities[neighbor] == curr_community)

	# Update the sums when moving the node
	sum_weights_C_new += node_degree
	sum_weights_C_old -= node_degree

	if node_color == 'red':
		sum_tot_R_new += node_degree
		sum_tot_R_old -= node_degree
		L_C_R_new += sum_intra_comm_new
		L_C_R_old -= sum_intra_comm_curr
	else:
		sum_tot_B_new += node_degree
		sum_tot_B_old -= node_degree
		L_C_B_new += sum_intra_comm_new
		L_C_B_old -= sum_intra_comm_curr
  
	# Compute new fairness modularity after moving the node
	Q_F_R_new = (L_C_R_new/m) - ((sum_weights_C_new * sum_tot_R_new)/(4 * m * m))
	Q_F_B_new = (L_C_B_new/m) - ((sum_weights_C_new * sum_tot_B_new)/(4 * m * m))
	Q_F_new = Q_F_R_new - Q_F_B_new
 
	# Compute change in fairness modularity
	delta_Q_F = Q_F_new - Q_F_old
	return delta_Q_F

# Calculate diversity fairness for a particular partition
def diversity_fairness(G, partition, color_dist, colors):
    n = G.number_of_nodes()
    m = G.size(weight="weight")

    color_list=color_dist.keys()
    
    # If n==0: return 0. Should not happen, but good failsafe
    if n==0: return 0

    sum_scores = 0.0
    F_dist = []
    # partition_summary = []
    
    # print("Partition in modularity_fairness function = ", partition)

    # Calculate modularity fairness for each community
    for i, ci in enumerate(partition):
        nodes_in_ci = [node for node in ci]
        # print("Nodes in Ci = ", nodes_in_ci)
        n_ci = len(nodes_in_ci)
        # print("Type of nodes_in_ci = ", type(nodes_in_ci[0]))
        
        if n_ci > 0:
            # Calculate number of intra-community edges with one node red and one node blue
            L_c_RB = sum(1 for u, v in G.edges(nodes_in_ci) if (colors[u] == "red" and colors[v] == "blue") or 
                         (colors[u] == "blue" and colors[v] == "red"))
            # Calculate sum of degress of all red and blue nodes
            d_c_R = sum(G.degree(u) for u in nodes_in_ci if colors[u] == "red")
            d_c_B = sum(G.degree(u) for u in nodes_in_ci if colors[u] == "blue")

            # Diversity fairness for community Ci
            diversity_fair_Ci = (L_c_RB - ((d_c_R * d_c_B)/m)) / (2 * m)
            weighted_diversity_fair_Ci = diversity_fair_Ci * n_ci/n

            sum_scores += weighted_diversity_fair_Ci
            F_dist.append(weighted_diversity_fair_Ci)

    return sum_scores, F_dist

# Helper function to calculate diversity modularity fairness gain when a node is moved to a new community
def diversity_fairness_gain(G, m, communities, node, new_community, community_edges_rb, community_red_degrees, community_blue_degrees, colors):
	node_color = colors[node] 	# red and blue, this is the only case defined in this algorithm
 
	curr_community = communities[node]

	# Calculate diversity before moving node
	L_C_RB_curr = community_edges_rb[curr_community]
	sum_tot_R_curr = community_red_degrees[curr_community]
	sum_tot_B_curr = community_blue_degrees[curr_community]
	diversity_RB_curr = (L_C_RB_curr - ((sum_tot_R_curr * sum_tot_B_curr)/m)) / (2 * m)
 
	# Calculate the number of intra-community edges with opposite color that have to be updated when node moves
	if node_color == "red":
		sum_intra_comm_new = sum(1 for neighbor in G.neighbors(node) if communities[neighbor] == new_community and colors[neighbor] == "blue")
		sum_intra_comm_curr = sum(1 for neighbor in G.neighbors(node) if communities[neighbor] == curr_community and colors[neighbor] == "blue")
	else:
		sum_intra_comm_new = sum(1 for neighbor in G.neighbors(node) if communities[neighbor] == new_community and colors[neighbor] == "red")
		sum_intra_comm_curr = sum(1 for neighbor in G.neighbors(node) if communities[neighbor] == curr_community and colors[neighbor] == "red")

	node_degree = G.degree(node)
	# Calculate diversity after moving node
	L_C_RB_new = community_edges_rb[new_community] + sum_intra_comm_new
	sum_tot_R_new = community_red_degrees[new_community]
	sum_tot_B_new = community_blue_degrees[new_community]
	if node_color == "red":
		sum_tot_R_new += node_degree
	else:
		sum_tot_B_new += node_degree
  
	diversity_RB_new = (L_C_RB_new - ((sum_tot_R_new * sum_tot_B_new)/m)) / (2 * m)
 
	# Compute change in fairness modularity
	delta_Q_F = diversity_RB_new - diversity_RB_curr
	return delta_Q_F


def computeDiversity(G, communities, weight="weight", resolution=1):
    out_degree = in_degree = dict(G.degree(weight=weight))
    deg_sum = sum(out_degree.values())
    m = deg_sum / 2
    norm = 1 / deg_sum**2
    
    deg_sum = sum(wt for u, v, wt in G.edges(data="inter_weight", default=1))
    
    mInter = deg_sum
    if deg_sum != 0:
        norm = 1 / (mInter*2*m)
    else:
        norm = 1
        
    degrees_red = dict(G.nodes(data="red_weight"))
    degrees_blue = dict(G.nodes(data="blue_weight"))
    
    def community_contribution(community):
        if len(community) > 0:
            comm = set(community)
            degree_R = sum(degrees_red[u] for u in comm)
            degree_B = sum(degrees_blue[u] for u in comm)
            inter_L = sum(wt for u, v, wt in G.edges(comm, data="inter_weight", default=1) if v in comm)
            modularityCommunityInter = (inter_L / (2*m)) - (resolution * degree_R * degree_B * norm)
            return modularityCommunityInter
        else:
            return 0
        
    communitiesNum = 0
    for c in communities:
        if len(c) > 0:
            communitiesNum += 1
            
    communityInterModularityList = []
    
    for community in communities:
        communityInterModularityList.append(community_contribution(community))
        
    return sum(communityInterModularityList), communityInterModularityList


def diversityMetricPaper(G, communities, colors, weight="weight", resolution=1):
	for u in G.nodes():
		G.nodes[u]["red_weight"] = 0
		G.nodes[u]["blue_weight"] = 0
		G.nodes[u]["inter_weight"] = 0
  
	for u, v in G.edges():
		G[u][v]["r_weight"] = 0
		G[u][v]["b_weight"] = 0
		G[u][v]["inter_weight"] = 0
  
	for u, v in G.edges():
		if colors[u] == "red" or colors[v] == "red":
			G[u][v]["r_weight"] = 1
		elif colors[u] == "blue" or colors[v] == "blue":
			G[u][v]["b_weight"] = 1
		else:
			G[u][v]["r_weight"] = 0
			G[u][v]["b_weight"] = 0
		if colors[u] != colors[v]:
			G[u][v]["inter_weight"] = 1
   
		if colors[u] == "blue" and colors[v] == "red":
			G.nodes[u]["red_weight"] += 1
			G.nodes[v]["blue_weight"] += 1
		if colors[u] == "red" and colors[v] == "blue":
			G.nodes[u]["blue_weight"] += 1
			G.nodes[v]["red_weight"] += 1
		if colors[u] != colors[v]:
			G.nodes[u]["inter_weight"] += 1
			G.nodes[v]["inter_weight"] += 1
   
	diversityModularity, diversityModularityDist = computeDiversity(G, communities, weight="weight", resolution=1)
   
	return diversityModularity, diversityModularityDist