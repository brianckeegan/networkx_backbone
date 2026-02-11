# networkx_backbone

Backbone extraction algorithms for complex networks, built on [NetworkX](https://networkx.org/).

This library provides 47 functions across 8 modules for extracting backbone
structures from weighted and unweighted networks.

## Installation

```bash
pip install networkx-backbone
```

For full functionality (required for statistical tests, bipartite methods, and some structural methods):

```bash
pip install networkx-backbone[full]
```

Or install from source:

```bash
git clone https://github.com/brianckeegan/networkx_backbone.git
cd networkx_backbone
pip install -e ".[full]"
```

## Modules

| Module | Description | Key Functions |
|--------|-------------|---------------|
| **statistical** | Hypothesis-testing methods | `disparity_filter`, `noise_corrected_filter`, `marginal_likelihood_filter`, `ecm_filter`, `lans_filter` |
| **structural** | Topology-based methods | `global_threshold_filter`, `strongest_n_ties`, `high_salience_skeleton`, `metric_backbone`, `ultrametric_backbone`, `doubly_stochastic_filter`, `h_backbone`, `modularity_backbone`, `planar_maximally_filtered_graph`, `maximum_spanning_tree_backbone` |
| **proximity** | Neighborhood-similarity scoring | `jaccard_backbone`, `dice_backbone`, `cosine_backbone`, `hub_promoted_index`, `hub_depressed_index`, `adamic_adar_index`, `resource_allocation_index`, `local_path_index`, and more |
| **hybrid** | Combined approaches | `glab_filter` |
| **bipartite** | Bipartite projection backbones | `sdsm`, `fdsm` |
| **unweighted** | Sparsification for unweighted graphs | `sparsify`, `lspar`, `local_degree` |
| **filters** | Post-hoc filtering utilities | `threshold_filter`, `fraction_filter`, `boolean_filter`, `consensus_backbone` |
| **measures** | Evaluation and comparison | `node_fraction`, `edge_fraction`, `weight_fraction`, `reachability`, `ks_degree`, `ks_weight`, `compare_backbones` |

## Quick Start

```python
import networkx as nx
import networkx_backbone as nb

# Create a weighted graph
G = nx.karate_club_graph()
for u, v in G.edges():
    G[u][v]["weight"] = 1.0

# Extract backbone using the disparity filter
H = nb.disparity_filter(G)

# Filter to keep only significant edges (p < 0.05)
backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05)

# Compare backbone to original
print(f"Edges kept: {nb.edge_fraction(G, backbone):.1%}")
print(f"Nodes kept: {nb.node_fraction(G, backbone):.1%}")
```

### Proximity-based scoring

```python
# Score edges by Jaccard similarity of endpoint neighborhoods
H = nb.jaccard_backbone(G)

# Keep only the top 20% most structurally embedded edges
backbone = nb.fraction_filter(H, "jaccard", 0.2, ascending=False)
```

### Bipartite backbone

```python
B = nx.Graph()
B.add_edges_from([(1, "a"), (1, "b"), (2, "a"), (2, "c"), (3, "b"), (3, "c")])

backbone = nb.sdsm(B, agent_nodes=[1, 2, 3], alpha=0.05)
```

### Comparing multiple methods

```python
backbones = {
    "disparity": nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05),
    "mst": nb.maximum_spanning_tree_backbone(G),
}
results = nb.compare_backbones(G, backbones)
```

## Dependencies

- **Required**: `networkx >= 3.0`
- **Optional**: `numpy >= 1.23`, `scipy >= 1.9` (needed for statistical methods, bipartite methods, doubly stochastic filter, local path index, and KS measures)

## Testing

```bash
pip install -e ".[test]"
pytest
```

## References

Key papers behind the implemented methods:

- Serrano, M. A., Boguna, M., & Vespignani, A. (2009). Extracting the multiscale backbone of complex weighted networks. *PNAS*, 106(16), 6483-6488.
- Coscia, M. & Neffke, F. M. (2017). Network backboning with noisy data. *Proc. IEEE ICDE*, 425-436.
- Grady, D., Thiemann, C., & Brockmann, D. (2012). Robust classification of salient links in complex networks. *Nature Communications*, 3, 864.
- Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance backbone of complex networks. *J. Complex Netw.*, 9, cnab021.
- Neal, Z. P. (2014). The backbone of bipartite projections. *Social Networks*, 39, 84-97.
- Neal, Z. P. (2022). backbone: An R package to extract network backbones. *PLOS ONE*, 17(5), e0269137.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
