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
from .fair_louvaines import (
	fair_louvain_communities
)

# sFairSC import
from ../ext_modules/sfairsc import (s_fair_sc)

# F-AL import
from ../ext_modules/fal import ()


for net in net_list:

	# Run n times
	for run_i in range(n_runs):

		# Run Fair-mod (balance) & MOUFLON (hybrid), a=[0.25,0.5,0.75]


		# Run sFairSC, k=(num clusters from experiments)


		# Run F-AL. Fairness=[group-mod, diversity]


		# keep stats for each algorithm: partition modularity, fairness (all defs?)
		#	num comms, performance. All with average and standard deviation