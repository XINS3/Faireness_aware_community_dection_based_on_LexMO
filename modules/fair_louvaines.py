import itertools
import math
from collections import deque

import networkx as nx
from networkx.algorithms.community import modularity
from networkx.utils import py_random_state

from .calc_partitions import (
                              _calculate_partition_diversity,
                              _calculate_partition_diversity_paper,
                              _calculate_partition_fexp,
                              _calculate_partition_fmody,
                              _calculate_partition_mod,
                              _calculate_partition_obj)
from .helpers import (_convert_multigraph, _gen_graph, diversity_fairness,
                      diversityMetricPaper, fairness_base, fairness_fexp,
                      modularity_fairness)


@py_random_state("seed")
def fair_louvain_communities(
	G, weight="weight", resolution=1, threshold=0.0000001, max_level=None, seed=None, color_list=["blue","red"], alpha=0.9, strategy="base"
):	
    partitions=[]

    # base strategy (fair-mod)
    if strategy=="base":
        partitions = fair_louvain_partitions_base(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

    # step2 strategy: optimize for modularity first, then switch to fairness
    elif strategy=="step2":
        partitions = fair_louvain_partitions_step2(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

    # fexp strategy: like base strategy, but also adds F_expected penalty
    elif strategy=="fexp":
        partitions = fair_louvain_partitions_fexp(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

    # hybrid strategy: combines step2 and fexp penalty
    elif strategy=="hybrid":
        partitions = fair_louvain_partitions_hybrid(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

    # modularity fairness strategy
    elif strategy=="fmody":
        partitions = fair_louvain_partitions_fmody(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)
    
    # diversity fairness strategy
    elif strategy=="diversity":
        # partitions = fair_louvain_partitions_diversity(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)
        partitions = fair_louvain_partitions_diversity_paper(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)
    
    # Step2 modularity fairness strategy: Combines step2 and fmody
    elif strategy=="step2fmody":
        partitions = fair_louvain_partitions_step2fmody(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)
    
    # Step2 diversity fairness strategy: Combines step2 and diversity
    elif strategy=="step2div":
        # partitions = fair_louvain_partitions_step2div(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)
        partitions = fair_louvain_partitions_step2div_paper(G, weight, resolution, threshold, seed, color_list=color_list, alpha=alpha)

    if max_level is not None:
        if max_level <= 0:
            raise ValueError("max_level argument must be a positive integer or None")
        partitions = itertools.islice(partitions, max_level)
    final_partition = deque(partitions, maxlen=1)
    return final_partition.pop()

### ------------ SELECT OPTIMIZATION STRATEGY ----------------

## Strategy "base"
## Base approach from Fair-mod paper: optimize the full Obj equation
@py_random_state("seed")
def fair_louvain_partitions_base(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]
	
	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	# Set n, m
	n = graph.number_of_nodes()
	m = graph.size(weight="weight")


	# Prepare partition colours
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

			# Set min_balance as the score. Normalize by comm size, times K-1
			balance_ci=(K_cols-1)*min_balance*n_ci/n

			# Get final score
			fscore_ci=balance_ci

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)

	
	# Run using full Obj
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_obj(
		graph, 
		n,
		m,
		partition,
		colors, 
		color_dist,
		partition_colors_new,
		phi,
		alpha=alpha,
		resolution=resolution,
		threshold=threshold, 
		is_directed=is_directed, 
		seed=seed
	)

	# Continue using full Obj=a*Q+(1-a)*F for improvements. Start with opt=0
	improvement = True
	opt=0
	while improvement:
		yield [s.copy() for s in partition]

		# Calculate new modularity, fairness and Obj scores
		new_mod = modularity(
			graph, inner_partition, resolution=resolution, weight="weight"
		)
		new_fair, _new_f_dist = fairness_base(
			G, partition, color_dist
		)
		new_opt = alpha * new_mod + (1-alpha) * new_fair

		# ...and stop optimizing if gain is less than threshold
		if new_opt - opt <= threshold:
			return

		mod = new_mod
		fair = new_fair
		opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_obj(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)

## Strategy "step2"
## Calculate partitions, first running only for modularity, and then optimizing for Q,F
@py_random_state("seed")
def fair_louvain_partitions_step2(G, weight="weight", resolution=1, threshold=0.00001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# Prepare partition colours
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

			# Set min_balance as the score. Normalize by comm size, times K-1
			balance_ci=(K_cols-1)*min_balance*n_ci/n

			# Get final score
			fscore_ci=balance_ci

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)

	# First step calculates only modularity gain
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_mod(
		graph, 
		n,
		m,
		partition, 
		colors,
		color_dist,
		phi,
		resolution=resolution, 
		is_directed=is_directed, 
		seed=seed
	)

	# Now start using full Obj=a*Q+(1-a)*F for improvements
	improvement = True
	first_fair_step = True
	while improvement:
		yield [s.copy() for s in partition]

		# For first step using fairness: calculate new round of improvement regardless
		if first_fair_step:
			# Set opt as previous (modularity only) optimum
			opt = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			# Remove flag
			first_fair_step=False

		# Otherwise check for improvement on Obj
		else:	
			new_mod = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			new_fair, _new_f_dist = fairness_base(
				G, partition, color_dist
			)

			new_opt = alpha * new_mod + (1-alpha) * new_fair

			if new_opt - opt <= threshold:
				return

			mod = new_mod
			fair = new_fair
			opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_obj(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)

## Strategy "fexp"
## Approach adding penalty for F_expected
@py_random_state("seed")
def fair_louvain_partitions_fexp(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)

	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	# Set n, m
	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# Prepare partition colors
	# Now calculate all colours for generated partitioning
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*n_ci/n)
			n_extra=n_ci-sum_dist

			# Calculate F_exp(c_i)
			f_exp=(K_cols*phi*n_ci+(phi+K_cols-1-(phi*K_cols))*n_extra)/((K_cols*n_ci + (phi-1)*n_extra))

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

			# Set min_balance as the score. Normalize by comm size, times K-1
			balance_ci=(K_cols-1)*min_balance#/len(comm)

			# Get final score
			if n_ci>=K_cols:
				fscore_ci=min(1.0,1-(f_exp-balance_ci))
			else:
				fscore_ci=0.0

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)

	# Run using full Obj
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_fexp(
		graph, 
		n,
		m,
		partition,
		colors, 
		color_dist,
		partition_colors_new,
		phi,
		alpha=alpha,
		resolution=resolution,
		threshold=threshold, 
		is_directed=is_directed, 
		seed=seed
	)

	# Continue using full Obj=a*Q+(1-a)*F for improvements. Start with opt=0
	improvement = True
	opt=0
	while improvement:
		yield [s.copy() for s in partition]

		# Calculate new modularity, fairness and Obj scores
		new_mod = modularity(
			graph, inner_partition, resolution=resolution, weight="weight"
		)
		new_fair, _new_f_dist = fairness_fexp(
			G, partition, color_dist
		)
		new_opt = alpha * new_mod + (1-alpha) * new_fair

		# # @DEBUG
		# print(f"------MAIN LOOP------")
		# print(f"opt={opt}, new_opt={new_opt} (new_mod={new_mod}, new_fair={new_fair})")

		# ...and stop optimizing if gain is less than threshold
		if new_opt - opt <= threshold:
			return

		mod = new_mod
		fair = new_fair
		opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_fexp(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)

## Strategy "hybrid"
## Hybrid approach: Step2 + penalty for F_expected
@py_random_state("seed")
def fair_louvain_partitions_hybrid(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)

	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		graph.add_nodes_from(G)
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	# Set n, m
	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# Prepare partition colors
	# Now calculate all colours for generated partitioning
	partition_colors_new=list()
	# For each community in partition:
	for comm in partition:
		n_ci=len(comm)
		if n_ci>0:
			# First, calculate extra nodes for the community
			sum_dist=0
			for col in color_list:
				sum_dist+=math.floor(color_dist[col]*n_ci/n)
			n_extra=n_ci-sum_dist

			# Calculate F_exp(c_i)
			f_exp=(K_cols*phi*n_ci+(phi+K_cols-1-(phi*K_cols))*n_extra) /((K_cols*n_ci + (phi-1)*n_extra))

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

			# Set min_balance as the score. Normalize by comm size, times K-1
			balance_ci=(K_cols-1)*min_balance#/len(comm)

			# Get final score
			if n_ci>=K_cols:
				fscore_ci=min(1.0,1-(f_exp-balance_ci))
			else:
				fscore_ci=0.0

		# Generate partition dict
		p_dict={}
		for col_ind,col in enumerate(color_list):
			p_dict[col]=sum_cols[col_ind]
		p_dict["score"]=fscore_ci

		# Append to list
		partition_colors_new.append(p_dict)
	
	# First step calculates only modularity gain
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_mod(
		graph,
		n, 
		m,
		partition, 
		colors,
		color_dist,
		phi,
		resolution=resolution, 
		is_directed=is_directed, 
		seed=seed,
		mode="fexp" # Set mode flag to calculate fairness score using F_exp
	)

	# Continue using full Obj=a*Q+(1-a)*F for improvements, with F_exp
	improvement = True
	first_fair_step = True
	opt=0
	while improvement:
		yield [s.copy() for s in partition]

		# # Calculate new modularity, fairness and Obj scores
		# new_mod = modularity(
		# 	graph, inner_partition, resolution=resolution, weight="weight"
		# )
		# new_fair, _new_f_dist = fairness_fexp(
		# 	G, partition, color_dist
		# )
		# new_opt = alpha * new_mod + (1-alpha) * new_fair


		# print(f"------MAIN LOOP------")
		# print(f"opt={opt}, new_opt={new_opt} (new_mod={new_mod}, new_fair={new_fair})")

		# For first step using fairness: calculate new round of improvement regardless
		if first_fair_step:
			# Set opt as previous (modularity only) optimum
			opt = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			# Remove flag
			first_fair_step=False

		# Otherwise check for improvement on Obj
		else:	
			new_mod = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			new_fair, _new_f_dist = fairness_fexp(
				G, partition, color_dist
			)

			new_opt = alpha * new_mod + (1-alpha) * new_fair

			if new_opt - opt <= threshold:
				return

			mod = new_mod
			fair = new_fair
			opt = new_opt

		# # ...and stop optimizing if gain is less than threshold
		# if new_opt - opt <= threshold:
		# 	return

		# mod = new_mod
		# fair = new_fair
		# opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Then continue using full Obj (with F_expected penalty)
		partition, inner_partition, improvement, partition_colors_new = _calculate_partition_fexp(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			partition_colors_new,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)
  
## Strategy "Fair Modularity (here fmody)"
## Approach: Build communities that "balances" connectivity of node colours within the community
@py_random_state("seed")
def fair_louvain_partitions_fmody(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
    partition = [{u} for u in G.nodes()]

    K_cols=len(color_list)
    # If one colour: stop. @TODO: revert to simple louvain?
    if K_cols==1:
        yield partition
        return

    colors=nx.get_node_attributes(G, "color")

    # Calculate network color ratios here to pass
    color_dist={}
    for c in color_list:
        color_dist[c]=0
    for n_ind in G.nodes():
        color_dist[colors[n_ind]]+=1

    # Also calculate phi=overall balance of colours in G
    c_least=min([color_dist[c] for c in color_dist])
    phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


    # If empty graph: return empty partition
    if nx.is_empty(G):
        yield partition
        return

    # Calculate partition modularity
    mod = modularity(G, partition, resolution=resolution, weight=weight)
    # print("Modularity score = ", mod)

    # Calculate partition modularity fairness score
    fmody_score, fmody_dist = modularity_fairness(G, partition, color_dist, colors)
    # print("Modularity Fairness score = ", fmody_score)
    # print("Modularity Fairness distribution = ", fmody_dist)
    
    # Calculate Q_dif_F
    Q_dif_F = (alpha * mod) + ((1-alpha) * fmody_score)
    # print("Q_dif_F = ", Q_dif_F) 

    # Convert multigraph if necessary
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        # graph = G.__class__()
        # # graph.add_nodes_from(G)
        # # Copy nodes with their attributes, including 'color'
        # for node, attrs in G.nodes(data=True):
        #     graph.add_node(node, **attrs)  # Copies all attributes
        # graph.add_weighted_edges_from(G.edges(data=weight, default=1))
        graph = G

    # Set n, m
    n = graph.number_of_nodes()
    m = graph.size(weight="weight")

    partition, inner_partition, improvement = _calculate_partition_fmody(
        graph,
        n,
        m,
        partition,
        colors,
        color_dist,
        phi,
        alpha=alpha,
        resolution=resolution,
        threshold=threshold,
        is_directed=is_directed,
        seed=seed
    )

    # Continue using full Obj=a*Q+(1-a)*F for improvements. Start with opt=0
    improvement = True
    opt = Q_dif_F   # Trying, maybe 0 itself would be better
    while improvement:
        yield [s.copy() for s in partition]
        
        # print("Partition = ", partition)

        # Calculate new modularity, fairness and Obj scores
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        new_fmody_score, _new_fmody_dist = modularity_fairness(
            G, partition, color_dist, colors
        )
        new_opt = alpha * new_mod + (1-alpha) * new_fmody_score
        # print("New opt = ", new_opt)

        # print("Difference = ", new_opt-opt)
        # ...and stop optimizing if gain is less than threshold
        if new_opt - opt <= threshold:
            return

        mod = new_mod
        fair = new_fmody_score
        opt = new_opt

        # Calculate new graph based on inner_partition
        graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)
        # print("new graph = ", graph)
        # print("new partition colors = ", partition_colors_new2)

        # Run for improvement again using full Obj
        partition, inner_partition, improvement = _calculate_partition_fmody(
            graph, 
            n,
            m,
            partition,
            colors, 
            color_dist,
            phi,
            alpha=alpha,
            resolution=resolution,
            threshold=threshold, 
            is_directed=is_directed, 
            seed=seed
        )
        
## Strategy "Diversity Modularity"
## Approach: Build communities that "balances" edge connectivity of node colours within the community
## This is done by checking if the members of sensitive attribute have a fair chance in taking part in link generation
## i.e. if every red node (Here red is sensitive) forms an edge with a blue node, its balanced and diversity balance will be 0
## Negative value corresponds to more blue nodes forming edges and positive value indicates more reds forming edges
@py_random_state("seed")
def fair_louvain_partitions_diversity(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
    partition = [{u} for u in G.nodes()]
    
    K_cols=len(color_list)
    # If one colour: stop. @TODO: revert to simple louvain?
    if K_cols==1:
        yield partition
        return

    colors=nx.get_node_attributes(G, "color")

    # Calculate network color ratios here to pass
    color_dist={}
    for c in color_list:
        color_dist[c]=0
    for n_ind in G.nodes():
        color_dist[colors[n_ind]]+=1

    # Also calculate phi=overall balance of colours in G
    c_least=min([color_dist[c] for c in color_dist])
    phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


    # If empty graph: return empty partition
    if nx.is_empty(G):
        yield partition
        return

    # Calculate partition modularity
    mod = modularity(G, partition, resolution=resolution, weight=weight)
    # print("Modularity score = ", mod)

    # Calculate partition modularity fairness score
    fmody_score, fmody_dist = diversity_fairness(G, partition, color_dist, colors)
    # print("Modularity Fairness score = ", fmody_score)
    # print("Modularity Fairness distribution = ", fmody_dist)
    
    # Calculate Q_dif_F
    Q_dif_F = (alpha * mod) + ((1-alpha) * fmody_score)
    # print("Q_dif_F = ", Q_dif_F) 

    # Convert multigraph if necessary
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        # graph.add_nodes_from(G)
        # Copy nodes with their attributes, including 'color'
        for node, attrs in G.nodes(data=True):
            graph.add_node(node, **attrs)  # Copies all attributes
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    # Set n, m
    n = graph.number_of_nodes()
    m = graph.size(weight="weight")

    partition, inner_partition, improvement = _calculate_partition_diversity(
        graph,
        n,
        m,
        partition,
        colors,
        color_dist,
        phi,
        alpha=alpha,
        resolution=resolution,
        threshold=threshold,
        is_directed=is_directed,
        seed=seed
    )        

    # Continue using full Obj=a*Q+(1-a)*F for improvements. Start with opt=0
    improvement = True
    opt = Q_dif_F   # Trying, maybe 0 itself would be better
    while improvement:
        yield [s.copy() for s in partition]
        
        # print("Partition = ", partition)

        # Calculate new modularity, fairness and Obj scores
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        new_fmody_score, _new_fmody_dist = diversity_fairness(
            G, partition, color_dist, colors
        )
        new_opt = alpha * new_mod + (1-alpha) * new_fmody_score
        # print("New opt = ", new_opt)

        # print("Difference = ", new_opt-opt)
        # ...and stop optimizing if gain is less than threshold
        if new_opt - opt <= threshold:
            return

        mod = new_mod
        fair = new_fmody_score
        opt = new_opt

        # Calculate new graph based on inner_partition
        graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)
        # print("new graph = ", graph)
        # print("new partition colors = ", partition_colors_new2)

        # Run for improvement again using full Obj
        partition, inner_partition, improvement = _calculate_partition_diversity(
            graph, 
            n,
            m,
            partition,
            colors, 
            color_dist,
            phi,
            alpha=alpha,
            resolution=resolution,
            threshold=threshold, 
            is_directed=is_directed, 
            seed=seed
        )
        
## Strategy "step2"
## Calculate partitions, first running only for modularity, and then optimizing for Q,F
@py_random_state("seed")
def fair_louvain_partitions_step2fmody(G, weight="weight", resolution=1, threshold=0.00001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		for node, attrs in G.nodes(data=True):
			graph.add_node(node, **attrs)  # Copies all attributes
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# First step calculates only modularity gain
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_mod(
		graph, 
		n,
		m,
		partition, 
		colors,
		color_dist,
		phi,
		resolution=resolution, 
		is_directed=is_directed, 
		seed=seed
	)

	# Now start using full Obj=a*Q+(1-a)*F for improvements
	improvement = True
	first_fair_step = True
	while improvement:
		yield [s.copy() for s in partition]

		# For first step using fairness: calculate new round of improvement regardless
		if first_fair_step:
			# Set opt as previous (modularity only) optimum
			opt = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			# Remove flag
			first_fair_step=False

		# Otherwise check for improvement on Obj
		else:	
			new_mod = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			new_fair, _new_f_dist = modularity_fairness(
				G, partition, color_dist, colors
			)

			new_opt = alpha * new_mod + (1-alpha) * new_fair

			if new_opt - opt <= threshold:
				return

			mod = new_mod
			fair = new_fair
			opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement = _calculate_partition_fmody(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)
  
## Strategy "step2div"
## Calculate partitions, first running only for modularity, and then optimizing for Diversity
@py_random_state("seed")
def fair_louvain_partitions_step2div(G, weight="weight", resolution=1, threshold=0.00001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)

	# Calculate partition modularity fairness score
	fmody_score, fmody_dist = diversity_fairness(G, partition, color_dist, colors)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		for node, attrs in G.nodes(data=True):
			graph.add_node(node, **attrs)  # Copies all attributes
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))

	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# First step calculates only modularity gain
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_mod(
		graph, 
		n,
		m,
		partition, 
		colors,
		color_dist,
		phi,
		resolution=resolution, 
		is_directed=is_directed, 
		seed=seed
	)

	# Now start using full Obj=a*Q+(1-a)*F for improvements
	improvement = True
	first_fair_step = True
	while improvement:
		yield [s.copy() for s in partition]

		# For first step using fairness: calculate new round of improvement regardless
		if first_fair_step:
			# Set opt as previous (modularity only) optimum
			opt = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			# Remove flag
			first_fair_step=False

		# Otherwise check for improvement on Obj
		else:	
			new_mod = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			new_fair, _new_f_dist = diversity_fairness(
				G, partition, color_dist, colors
			)

			new_opt = alpha * new_mod + (1-alpha) * new_fair

			if new_opt - opt <= threshold:
				return

			mod = new_mod
			fair = new_fair
			opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Run for improvement again using full Obj
		partition, inner_partition, improvement = _calculate_partition_diversity(
			graph, 
			n,
			m,
			partition,
			colors, 
			color_dist,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)

# Diversity Modularity based on the paper 
@py_random_state("seed")
def fair_louvain_partitions_diversity_paper(G, weight="weight", resolution=1, threshold=0.000001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)
	# print("Modularity score = ", mod)

	# Calculate partition modularity fairness score
	div_score, div_dist = diversityMetricPaper(G, partition, colors, weight=weight, resolution=resolution)
	# print("Modularity Fairness score = ", fmody_score)
	# print("Modularity Fairness distribution = ", fmody_dist)

	# Calculate Q_dif_F
	Q_dif_F = (alpha * mod) + ((1-alpha) * div_score)
	# print("Q_dif_F = ", Q_dif_F) 

	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		# graph.add_nodes_from(G)
		# Copy nodes with their attributes, including 'color'
		for node, attrs in G.nodes(data=True):
			graph.add_node(node, **attrs)  # Copies all attributes
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))
		
		# Add the node attributes for the red and blue weights
		for u in graph.nodes():
			graph.nodes[u]["red_weight"] = 0
			graph.nodes[u]["blue_weight"] = 0
			graph.nodes[u]["inter_weight"] = 0
		for u, v in graph.edges():
			graph[u][v]["r_weight"] = 0
			graph[u][v]["b_weight"] = 0
			graph[u][v]["inter_weight"] = 0

		for u, v in graph.edges():
			if colors[u] == "red" or colors[v] == "red":
				graph[u][v]["r_weight"] = 1
				graph[u][v]["b_weight"] = 0
			elif colors[u] == "blue" or colors[v] == "blue":
				graph[u][v]["r_weight"] = 0
				graph[u][v]["b_weight"] = 1

			if colors[u] != colors[v]:
				graph[u][v]["inter_weight"] = 1

			if colors[v] == "red":
				graph.nodes[u]["red_weight"] += 1
			if colors[u] == "red":
				graph.nodes[v]["red_weight"] += 1
			if colors[v] == "blue":
				graph.nodes[u]["blue_weight"] += 1
			if colors[u] == "blue":
				graph.nodes[v]["blue_weight"] += 1
			if colors[v] != colors[u]:
				graph.nodes[u]["inter_weight"] += 1
				graph.nodes[v]["inter_weight"] += 1

	# Set n, m
	n = graph.number_of_nodes()
	m = graph.size(weight="weight")
	iterationNum = 0
 
	# Count the number of neighbors with r_weight equal to 1 for each node
	r_weight_neighbors_count = {}
	for node in graph.nodes():
		r_weight_neighbors_count[node] = sum(1 for neighbor in graph.neighbors(node) if graph[node][neighbor].get('inter_weight') == 1)

	# Sort nodes based on the number of neighbors with r_weight equal to 1
	sorted_nodes = sorted(graph.nodes(), key=lambda node: r_weight_neighbors_count[node], reverse=True)
	nodesList = list(sorted_nodes)

	partition, inner_partition, improvement = _calculate_partition_diversity_paper(
		graph,
		n,
		m,
		iterationNum,
		partition,
		div_dist,
		nodesList,
		colors,
		phi,
		alpha=alpha,
		resolution=resolution,
		threshold=threshold,
		is_directed=is_directed,
		seed=seed
	)

	# Continue using full Obj=a*Q+(1-a)*F for improvements. Start with opt=0
	improvement = True
	iterationNum = 1
	opt = Q_dif_F   # Trying, maybe 0 itself would be better
	while improvement:
		yield [s.copy() for s in partition]
		
		# print("Partition = ", partition)

		# Calculate new modularity, fairness and Obj scores
		new_mod = modularity(
			graph, inner_partition, resolution=resolution, weight="weight"
		)
		new_div_score, new_div_dist = diversityMetricPaper(
			graph, inner_partition, colors, weight="weight", resolution=resolution
		)
		new_opt = alpha * new_mod + (1-alpha) * new_div_score
		# print("New opt = ", new_opt)

		# print("Difference = ", new_opt-opt)
		# ...and stop optimizing if gain is less than threshold
		if new_opt - opt <= threshold:
			return

		mod = new_mod
		fair = new_div_score
		opt = new_opt

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors, diversity_flag=True)
		# print("new graph = ", graph)
		# print("new partition colors = ", partition_colors_new2)
  
		# Count the number of neighbors with r_weight equal to 1 for each node
		r_weight_neighbors_count = {}
		for node in graph.nodes():
			r_weight_neighbors_count[node] = sum(1 for neighbor in graph.neighbors(node) if graph[node][neighbor].get('inter_weight') == 1)

		# Sort nodes based on the number of neighbors with r_weight equal to 1
		sorted_nodes = sorted(graph.nodes(), key=lambda node: r_weight_neighbors_count[node], reverse=True)
		nodesList = list(sorted_nodes)

		# Run for improvement again using full Obj
		partition, inner_partition, improvement = _calculate_partition_diversity_paper(
			graph, 
			n,
			m,
			iterationNum,
			partition,
			new_div_dist,
			nodesList,
			colors,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)
		iterationNum += 1

# Step2div_based on Diversity in the paper  
@py_random_state("seed")
def fair_louvain_partitions_step2div_paper(G, weight="weight", resolution=1, threshold=0.00001, seed=None, color_list=["blue","red"], alpha=0.9):
	partition = [{u} for u in G.nodes()]

	K_cols=len(color_list)
	# If one colour: stop. @TODO: revert to simple louvain?
	if K_cols==1:
		yield partition
		return

	colors=nx.get_node_attributes(G, "color")

	# Calculate network color ratios here to pass
	color_dist={}
	for c in color_list:
		color_dist[c]=0
	for n_ind in G.nodes():
		color_dist[colors[n_ind]]+=1

	# Also calculate phi=overall balance of colours in G
	c_least=min([color_dist[c] for c in color_dist])
	phi=(K_cols-1)*c_least/(len(G.nodes())-c_least)


	# If empty graph: return empty partition
	if nx.is_empty(G):
		yield partition
		return

	# Calculate partition modularity
	mod = modularity(G, partition, resolution=resolution, weight=weight)

	# Calculate partition modularity fairness score
	div_score, div_dist = diversityMetricPaper(G, partition, colors, weight=weight, resolution=resolution)
	
	# Convert multigraph if necessary
	is_directed = G.is_directed()
	if G.is_multigraph():
		graph = _convert_multigraph(G, weight, is_directed)
	else:
		graph = G.__class__()
		for node, attrs in G.nodes(data=True):
			graph.add_node(node, **attrs)  # Copies all attributes
		graph.add_weighted_edges_from(G.edges(data=weight, default=1))
		
		# Add the node attributes for the red and blue weights
		for u in graph.nodes():
			graph.nodes[u]["red_weight"] = 0
			graph.nodes[u]["blue_weight"] = 0
			graph.nodes[u]["inter_weight"] = 0
		for u, v in graph.edges():
			graph[u][v]["r_weight"] = 0
			graph[u][v]["b_weight"] = 0
			graph[u][v]["inter_weight"] = 0

		for u, v in graph.edges():
			if colors[u] == "red" or colors[v] == "red":
				graph[u][v]["r_weight"] = 1
				graph[u][v]["b_weight"] = 0
			elif colors[u] == "blue" or colors[v] == "blue":
				graph[u][v]["r_weight"] = 0
				graph[u][v]["b_weight"] = 1

			if colors[u] != colors[v]:
				graph[u][v]["inter_weight"] = 1

			if colors[v] == "red":
				graph.nodes[u]["red_weight"] += 1
			if colors[u] == "red":
				graph.nodes[v]["red_weight"] += 1
			if colors[v] == "blue":
				graph.nodes[u]["blue_weight"] += 1
			if colors[u] == "blue":
				graph.nodes[v]["blue_weight"] += 1
			if colors[v] != colors[u]:
				graph.nodes[u]["inter_weight"] += 1
				graph.nodes[v]["inter_weight"] += 1

	n = graph.number_of_nodes()
	m = graph.size(weight="weight")

	# First step calculates only modularity gain
	partition, inner_partition, improvement, partition_colors_new = _calculate_partition_mod(
		graph, 
		n,
		m,
		partition, 
		colors,
		color_dist,
		phi,
		resolution=resolution, 
		is_directed=is_directed, 
		seed=seed
	)
 
	div, div_dist = diversityMetricPaper(
		graph, partition, colors, weight="weight", resolution=resolution
	)

	# Now start using full Obj=a*Q+(1-a)*F for improvements
	improvement = True
	first_fair_step = True
	iterationNum = 1
	while improvement:
		yield [s.copy() for s in partition]

		# For first step using fairness: calculate new round of improvement regardless
		if first_fair_step:
			# Set opt as previous (modularity only) optimum
			opt = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			# Remove flag
			first_fair_step=False

		# Otherwise check for improvement on Obj
		else:	
			new_mod = modularity(
				graph, inner_partition, resolution=resolution, weight="weight"
			)
			new_div, new_div_dist = diversityMetricPaper(
				graph, inner_partition, colors, weight="weight", resolution=resolution
			)

			new_opt = alpha * new_mod + (1-alpha) * new_div

			if new_opt - opt <= threshold:
				return

			mod = new_mod
			fair = new_div
			opt = new_opt
			div_dist = new_div_dist

		# Calculate new graph based on inner_partition
		graph, partition_colors_new2 = _gen_graph(graph, inner_partition, colors, diversity_flag=True)

		# Refresh partition colors
		partition_colors = partition_colors_new

		# Count the number of neighbors with r_weight equal to 1 for each node
		r_weight_neighbors_count = {}
		for node in graph.nodes():
			r_weight_neighbors_count[node] = sum(1 for neighbor in graph.neighbors(node) if graph[node][neighbor].get('inter_weight') == 1)

		# Sort nodes based on the number of neighbors with r_weight equal to 1
		sorted_nodes = sorted(graph.nodes(), key=lambda node: r_weight_neighbors_count[node], reverse=True)
		nodesList = list(sorted_nodes)

		# Run for improvement again using full Obj
		partition, inner_partition, improvement = _calculate_partition_diversity_paper(
			graph, 
			n,
			m,
			iterationNum,
			partition,
			div_dist,
			nodesList,
			colors,
			phi,
			alpha=alpha,
			resolution=resolution,
			threshold=threshold, 
			is_directed=is_directed, 
			seed=seed
		)
		iterationNum += 1