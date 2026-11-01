"""Methods-section text and citations for a backbone.

:func:`~networkx_backbone.describe_backbone` is the counterpart of the
``narrative`` option of Neal's *Backbone 3.0* R package: it returns a short methods paragraph naming
the method and its null model, the parameters, the multiple-testing
correction, how much of the network was kept, and the papers to cite
(including this package and its version).

The text is built from one registry, :data:`METHODS`, which maps the name of
every public scoring or extraction function to a one-sentence description and
the keys of its references in :data:`REFERENCES` (taken from the functions'
own *References* sections).  The coverage policy is:

- **Described**: every public function of the ``statistical``,
  ``structural``, ``proximity``, ``hybrid``, ``bipartite``, ``hypergraph``
  and ``unweighted`` modules that scores edges (or nodes, pairs or
  hyperedges) or extracts a backbone, including the bipartite projections.
  Short aliases such as ``disparity`` or ``svh`` and the method names
  accepted by the dispatch wrappers (``"global"``, ``"simple"``, ...) are
  resolved through :data:`ALIASES`.
- **Not described** (:data:`EXCLUDED`, with the reason for each): the
  dispatch wrappers (describe the method they dispatch to), null-model
  building blocks (``bicm``, ``fastball``), hypergraph helpers that are not
  backbones, and result classes.
- The helper modules ``filters``, ``measures``, ``hypergraph_io``,
  ``visualization``, ``describe`` and ``convert`` are not methods and are
  skipped.

The test suite checks that every public name falls in one of these groups, so
a new method cannot be added without a registry entry.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from networkx_backbone._typing import Hyperedges
    from networkx_backbone.hypergraph import HypergraphBackbone, ValidatedHypergraph

__all__ = ["describe_backbone"]

PACKAGE = "networkx-backbone"
REPOSITORY = "https://github.com/brianckeegan/networkx_backbone"

#: Modules whose public functions are helpers rather than backbone methods.
HELPER_MODULES = frozenset(
    {"filters", "measures", "hypergraph_io", "visualization", "describe", "convert"}
)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """A bibliography entry, rendered as an in-text citation, text or BibTeX.

    ``kind`` is the BibTeX entry type (``"article"``, ``"inproceedings"``,
    ``"book"`` or ``"misc"``); ``venue`` is the journal, proceedings or
    publisher.  ``eprint`` is an arXiv identifier.
    """

    key: str
    kind: str
    authors: tuple
    year: int
    title: str
    venue: str = ""
    volume: str = ""
    number: str = ""
    pages: str = ""
    doi: str = ""
    eprint: str = ""

    def cite(self):
        """In-text citation such as ``"Serrano et al., 2009"``."""
        names = [a.split(",")[0] for a in self.authors]
        if len(names) == 1:
            who = names[0]
        elif len(names) == 2:
            who = f"{names[0]} & {names[1]}"
        else:
            who = f"{names[0]} et al."
        return f"{who}, {self.year}"

    def text(self, markdown=False):
        """Full reference in APA-like style."""
        authors = list(self.authors)
        if len(authors) > 1:
            who = ", ".join(authors[:-1]) + ", & " + authors[-1]
        else:
            who = authors[0]
        parts = [f"{who} ({self.year}). {self.title}."]
        venue = f"*{self.venue}*" if markdown and self.venue else self.venue
        if self.kind == "misc" and self.eprint:
            parts.append(f"arXiv:{self.eprint}.")
        elif venue:
            detail = venue
            if self.volume:
                vol = f"*{self.volume}*" if markdown else self.volume
                detail += f", {vol}"
                if self.number:
                    detail += f"({self.number})"
            if self.pages:
                detail += f", {self.pages.replace('--', '-')}"
            parts.append(detail + ".")
        if self.doi:
            url = f"https://doi.org/{self.doi}"
            parts.append(f"<{url}>" if markdown else url)
        return " ".join(parts)

    def bibtex(self):
        """BibTeX entry."""
        field_name = {"article": "journal", "inproceedings": "booktitle"}.get(
            self.kind, "publisher"
        )
        fields = [
            ("author", " and ".join(self.authors)),
            ("title", self.title),
            (field_name, "" if self.kind == "misc" else self.venue),
            ("volume", self.volume),
            ("number", self.number),
            ("pages", self.pages),
            ("year", str(self.year)),
            ("doi", self.doi),
        ]
        if self.eprint:
            fields += [("eprint", self.eprint), ("archiveprefix", "arXiv")]
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
        return f"@{self.kind}{{{self.key},\n{body}\n}}"


def _refs(*entries):
    return {r.key: r for r in entries}


#: Bibliography of every method in :data:`METHODS` and every correction in
#: :func:`~networkx_backbone.adjust_pvalues`, keyed by BibTeX key.
REFERENCES = _refs(
    # --- statistical -------------------------------------------------------
    Reference(
        "serrano2009", "article",
        ("Serrano, M. A.", "Boguñá, M.", "Vespignani, A."), 2009,
        "Extracting the multiscale backbone of complex weighted networks",
        "Proceedings of the National Academy of Sciences", "106", "16",
        "6483--6488", doi="10.1073/pnas.0808904106",
    ),
    Reference(
        "coscia2017", "inproceedings", ("Coscia, M.", "Neffke, F. M."), 2017,
        "Network backboning with noisy data",
        "Proceedings of the IEEE International Conference on Data Engineering "
        "(ICDE)", pages="425--436", doi="10.1109/ICDE.2017.100",
        eprint="1701.07336",
    ),
    Reference(
        "coscia2019", "inproceedings", ("Coscia, M.", "Rossi, L."), 2019,
        "The impact of projection and backboning on network topologies",
        "Proceedings of the IEEE/ACM International Conference on Advances in "
        "Social Networks Analysis and Mining (ASONAM)",
        doi="10.1145/3341161.3342862", eprint="1906.09081",
    ),
    Reference(
        "newman2001a", "article", ("Newman, M. E. J.",), 2001,
        "Scientific collaboration networks. I. Network construction and "
        "fundamental results", "Physical Review E", "64", pages="016131",
    ),
    Reference(
        "newman2001b", "article", ("Newman, M. E. J.",), 2001,
        "Scientific collaboration networks. II. Shortest paths, weighted "
        "networks, and centrality", "Physical Review E", "64", pages="016132",
    ),
    Reference(
        "zhou2007", "article",
        ("Zhou, T.", "Ren, J.", "Medo, M.", "Zhang, Y.-C."), 2007,
        "Bipartite network projection and personal recommendation",
        "Physical Review E", "76", pages="046115",
        doi="10.1103/PhysRevE.76.046115",
    ),
    Reference(
        "yildirim2014", "article", ("Yildirim, M. A.", "Coscia, M."), 2014,
        "Using random walks to generate associations between objects",
        "PLoS ONE", "9", "8", "e104813", doi="10.1371/journal.pone.0104813",
    ),
    Reference(
        "dianati2016", "article", ("Dianati, N.",), 2016,
        "Unwinding the hairball graph: Pruning algorithms for weighted "
        "complex networks", "Physical Review E", "93", pages="012304",
        doi="10.1103/PhysRevE.93.012304",
    ),
    Reference(
        "gemmetto2017", "misc",
        ("Gemmetto, V.", "Cardillo, A.", "Garlaschelli, D."), 2017,
        "Irreducible network backbones: unbiased graph filtering via maximum "
        "entropy", eprint="1706.00230",
    ),
    Reference(
        "marcaccioli2019", "article", ("Marcaccioli, R.", "Livan, G."), 2019,
        "A Pólya urn approach to information filtering in complex networks",
        "Nature Communications", "10", pages="745",
        doi="10.1038/s41467-019-08667-3",
    ),
    Reference(
        "foti2011", "article",
        ("Foti, N. J.", "Hughes, J. M.", "Rockmore, D. N."), 2011,
        "Nonparametric sparsification of complex multiscale networks",
        "PLoS ONE", "6", "2", "e16431", doi="10.1371/journal.pone.0016431",
    ),
    Reference(
        "radicchi2011", "article",
        ("Radicchi, F.", "Ramasco, J. J.", "Fortunato, S."), 2011,
        "Information filtering in complex weighted networks",
        "Physical Review E", "83", "4", "046101",
        doi="10.1103/PhysRevE.83.046101",
    ),
    Reference(
        "vannuffel2007", "article", ("Van Nuffel, N.",), 2007,
        "Determination of the number of significant flows in "
        "origin-destination specific analysis: the case of commuting in "
        "Flanders", "Regional Studies", "41", "4", "509--524",
        doi="10.1080/00343400701281808",
    ),
    Reference(
        "vannuffel2010", "article",
        ("Van Nuffel, N.", "Derudder, B.", "Witlox, F."), 2010,
        "Even important connections are not always meaningful: on the use "
        "of a polarisation measure in a typology of European cities in air "
        "transport networks", "Tijdschrift voor Economische en Sociale "
        "Geografie", "101", "3", "333--348",
        doi="10.1111/j.1467-9663.2009.00547.x",
    ),
    Reference(
        "weaver1954", "article", ("Weaver, J. C.",), 1954,
        "Crop-combination regions in the Middle West", "Geographical Review",
        "44", "2", "175--200", doi="10.2307/212355",
    ),
    Reference(
        "yassin2023", "article",
        ("Yassin, A.", "Haidar, A.", "Cherifi, H.", "Seba, H.", "Togni, O."),
        2023,
        "An evaluation tool for backbone extraction techniques in weighted "
        "complex networks", "Scientific Reports", "13", pages="17000",
        doi="10.1038/s41598-023-42076-3",
    ),
    # --- structural --------------------------------------------------------
    Reference(
        "satuluri2011", "inproceedings",
        ("Satuluri, V.", "Parthasarathy, S.", "Ruan, Y."), 2011,
        "Local graph sparsification for scalable clustering",
        "Proceedings of the ACM SIGMOD International Conference on Management "
        "of Data", pages="721--732", doi="10.1145/1989323.1989399",
    ),
    Reference(
        "nystuen1961", "article", ("Nystuen, J. D.", "Dacey, M. F."), 1961,
        "A graph theory interpretation of nodal regions",
        "Papers in Regional Science", "7", "1", "29--42",
    ),
    Reference(
        "girvan2002", "article", ("Girvan, M.", "Newman, M. E. J."), 2002,
        "Community structure in social and biological networks",
        "Proceedings of the National Academy of Sciences", "99", "12",
        "7821--7826", doi="10.1073/pnas.122653799",
    ),
    Reference(
        "freeman1978", "article", ("Freeman, L. C.",), 1978,
        "Centrality in social networks: conceptual clarification",
        "Social Networks", "1", "3", "215--239",
    ),
    Reference(
        "grady2012", "article",
        ("Grady, D.", "Thiemann, C.", "Brockmann, D."), 2012,
        "Robust classification of salient links in complex networks",
        "Nature Communications", "3", pages="864", doi="10.1038/ncomms1847",
    ),
    Reference(
        "simas2021", "article",
        ("Simas, T.", "Correia, R. B.", "Rocha, L. M."), 2021,
        "The distance backbone of complex networks",
        "Journal of Complex Networks", "9", "6", "cnab021",
        doi="10.1093/comnet/cnab021",
    ),
    Reference(
        "slater2009", "article", ("Slater, P. B.",), 2009,
        "A two-stage algorithm for extracting the multiscale backbone of "
        "complex weighted networks",
        "Proceedings of the National Academy of Sciences", "106", "26", "E66",
    ),
    Reference(
        "zhang2018", "article",
        ("Zhang, R. J.", "Stanley, H. E.", "Ye, F. Y."), 2018,
        "Extracting h-backbone as a core structure in weighted networks",
        "Scientific Reports", "8", pages="14356",
    ),
    Reference(
        "rajeh2022", "inproceedings",
        ("Rajeh, S.", "Savonnet, M.", "Leclercq, E.", "Cherifi, H."), 2022,
        "Modularity-based backbone extraction in weighted complex networks",
        "NetSci-X 2022", pages="67--79",
    ),
    Reference(
        "tumminello2005", "article",
        ("Tumminello, M.", "Aste, T.", "Di Matteo, T.", "Mantegna, R. N."),
        2005, "A tool for filtering information in complex systems",
        "Proceedings of the National Academy of Sciences", "102", "30",
        "10421--10426",
    ),
    # --- proximity ---------------------------------------------------------
    Reference(
        "jaccard1901", "article", ("Jaccard, P.",), 1901,
        "Distribution de la flore alpine dans le bassin des Dranses et dans "
        "quelques regions voisines",
        "Bulletin de la Societe Vaudoise des Sciences Naturelles", "37",
        pages="241--272",
    ),
    Reference(
        "dice1945", "article", ("Dice, L. R.",), 1945,
        "Measures of the amount of ecologic association between species",
        "Ecology", "26", "3", "297--302",
    ),
    Reference(
        "salton1983", "book", ("Salton, G.", "McGill, M. J."), 1983,
        "Introduction to Modern Information Retrieval", "McGraw-Hill",
    ),
    Reference(
        "ravasz2002", "article",
        ("Ravasz, E.", "Somera, A. L.", "Mongru, D. A.", "Oltvai, Z. N.",
         "Barabási, A.-L."), 2002,
        "Hierarchical organization of modularity in metabolic networks",
        "Science", "297", "5586", "1551--1555",
    ),
    Reference(
        "leicht2006", "article",
        ("Leicht, E. A.", "Holme, P.", "Newman, M. E. J."), 2006,
        "Vertex similarity in networks", "Physical Review E", "73", "2",
        "026120",
    ),
    Reference(
        "barabasi1999", "article", ("Barabási, A.-L.", "Albert, R."), 1999,
        "Emergence of scaling in random networks", "Science", "286", "5439",
        "509--512",
    ),
    Reference(
        "adamic2003", "article", ("Adamic, L. A.", "Adar, E."), 2003,
        "Friends and neighbors on the Web", "Social Networks", "25", "3",
        "211--230",
    ),
    Reference(
        "zhou2009", "article", ("Zhou, T.", "Lü, L.", "Zhang, Y.-C."), 2009,
        "Predicting missing links via local information",
        "European Physical Journal B", "71", "4", "623--630",
    ),
    Reference(
        "libennowell2007", "article",
        ("Liben-Nowell, D.", "Kleinberg, J."), 2007,
        "The link-prediction problem for social networks",
        "Journal of the American Society for Information Science and "
        "Technology", "58", "7", "1019--1031",
    ),
    Reference(
        "lu2009", "article", ("Lü, L.", "Jin, C.-H.", "Zhou, T."), 2009,
        "Similarity index based on local paths for link prediction of "
        "complex networks", "Physical Review E", "80", "4", "046122",
    ),
    # --- hybrid ------------------------------------------------------------
    Reference(
        "zhang2014", "article",
        ("Zhang, X.", "Zhang, Z.", "Zhao, H.", "Wang, Q.", "Zhu, J."), 2014,
        "Extracting the globally and locally adaptive backbone of complex "
        "networks", "PLoS ONE", "9", "6", "e100428",
        doi="10.1371/journal.pone.0100428",
    ),
    # --- bipartite ---------------------------------------------------------
    Reference(
        "neal2014", "article", ("Neal, Z. P.",), 2014,
        "The backbone of bipartite projections: Inferring relationships from "
        "co-authorship, co-sponsorship, co-attendance and other co-behaviors",
        "Social Networks", "39", pages="84--97",
        doi="10.1016/j.socnet.2014.06.001",
    ),
    Reference(
        "neal2023", "inproceedings", ("Neal, Z. P.", "Neal, J. W."), 2023,
        "Stochastic Degree Sequence Model with Edge Constraints (SDSM-EC) for "
        "Backbone Extraction", "Complex Networks XII", pages="127--136",
        doi="10.1007/978-3-031-53468-3_11",
    ),
    Reference(
        "neal2021", "article",
        ("Neal, Z. P.", "Domagalski, R.", "Sagan, B."), 2021,
        "Comparing alternatives to the fixed degree sequence model for "
        "extracting the backbone of bipartite projections",
        "Scientific Reports", "11", pages="23929",
        doi="10.1038/s41598-021-03238-3",
    ),
    Reference(
        "godard2022", "article", ("Godard, K.", "Neal, Z. P."), 2022,
        "fastball: A fast algorithm to sample bipartite graphs with fixed "
        "degree sequences", "Journal of Complex Networks", "10", "6",
        "cnac049", doi="10.1093/comnet/cnac049",
    ),
    # --- hypergraph --------------------------------------------------------
    Reference(
        "kirkley2026", "misc",
        ("Kirkley, A.", "Felippe, H.", "Malizia, F.", "Battiston, F."), 2026,
        "Hypergraph backboning", eprint="2606.00893",
    ),
    Reference(
        "musciotto2021", "article",
        ("Musciotto, F.", "Battiston, F.", "Mantegna, R. N."), 2021,
        "Detecting informative higher-order interactions in statistically "
        "validated hypergraphs", "Communications Physics", "4", pages="218",
    ),
    # --- unweighted --------------------------------------------------------
    Reference(
        "neal2022", "article", ("Neal, Z. P.",), 2022,
        "backbone: An R package to extract network backbones", "PLoS ONE",
        "17", "5", "e0269137", doi="10.1371/journal.pone.0269137",
    ),
    Reference(
        "lindner2015", "inproceedings",
        ("Lindner, G.", "Staudt, C. L.", "Hamann, M.", "Meyerhenke, H.",
         "Wagner, D."), 2015,
        "Structure-preserving sparsification of social networks",
        "Proceedings of the 2015 IEEE/ACM International Conference on "
        "Advances in Social Networks Analysis and Mining (ASONAM)",
        pages="448--454",
    ),
    Reference(
        "hamann2016", "article",
        ("Hamann, M.", "Lindner, G.", "Meyerhenke, H.", "Staudt, C. L.",
         "Wagner, D."), 2016,
        "Structure-preserving sparsification methods for social networks",
        "Social Network Analysis and Mining", "6", "1", "22",
        doi="10.1007/s13278-016-0332-2",
    ),
    # --- multiple-testing corrections ---------------------------------------
    Reference(
        "holm1979", "article", ("Holm, S.",), 1979,
        "A simple sequentially rejective multiple test procedure",
        "Scandinavian Journal of Statistics", "6", "2", "65--70",
    ),
    Reference(
        "hochberg1988", "article", ("Hochberg, Y.",), 1988,
        "A sharper Bonferroni procedure for multiple tests of significance",
        "Biometrika", "75", "4", "800--802",
    ),
    Reference(
        "benjamini1995", "article", ("Benjamini, Y.", "Hochberg, Y."), 1995,
        "Controlling the false discovery rate: a practical and powerful "
        "approach to multiple testing",
        "Journal of the Royal Statistical Society: Series B", "57", "1",
        "289--300",
    ),
    Reference(
        "benjamini2001", "article", ("Benjamini, Y.", "Yekutieli, D."), 2001,
        "The control of the false discovery rate in multiple testing under "
        "dependency", "Annals of Statistics", "29", "4", "1165--1188",
    ),
)  # fmt: skip


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    """Registry entry for one backbone method.

    label
        Name of the method as it reads in a sentence ("the disparity filter").
    summary
        Relative clause describing what the method scores and against which
        null model or criterion ("which tests ...").
    refs
        Keys into :data:`REFERENCES`.
    kind
        Input the method takes: ``"weighted"``, ``"unweighted"``,
        ``"bipartite"`` (agent-pair projection) or ``"hypergraph"``.
    test
        Whether the method yields p-values to compare with ``alpha``.
    builtin_mtc
        Correction the method applies itself (``"bh"`` for SVH/SVC), in
        which case *mtc* must stay ``"none"``.
    """

    label: str
    summary: str
    refs: tuple = ()
    kind: str = "weighted"
    test: bool = False
    builtin_mtc: str | None = None


W, U, B, H = "weighted", "unweighted", "bipartite", "hypergraph"

_SHARED = "scores each edge by the number of neighbors its endpoints share"


def _overlap(normalizer):
    return f"{_SHARED}, divided by {normalizer}"


#: Registry of every described method: function name -> :class:`Method`.
METHODS = {
    # --- statistical -------------------------------------------------------
    "disparity_filter": Method(
        "the disparity filter",
        "which tests each edge's weight against a null model in which each "
        "node's strength is spread uniformly at random over its edges",
        ("serrano2009",), test=True,
    ),
    "noise_corrected_filter": Method(
        "the noise-corrected filter",
        "which compares each edge's weight with the value expected from its "
        "endpoints' strengths and tests the resulting lift against its "
        "posterior standard deviation under a Bayesian binomial null model",
        ("coscia2017",), test=True,
    ),
    "marginal_likelihood_filter": Method(
        "the marginal likelihood filter",
        "which tests each edge's weight against a binomial null model in which "
        "unit edges choose their endpoints with probability proportional to "
        "node strength",
        ("dianati2016",), test=True,
    ),
    "ecm_filter": Method(
        "the enhanced configuration model (ECM) filter",
        "which tests each edge's weight against the maximum-entropy ensemble "
        "of weighted graphs that preserves every node's degree and strength "
        "in expectation",
        ("gemmetto2017",), test=True,
    ),
    "lans_filter": Method(
        "locally adaptive network sparsification (LANS)",
        "which tests each edge's weight nonparametrically against the "
        "empirical distribution of the weights incident to its endpoints",
        ("foti2011",), test=True,
    ),
    "multiple_linkage_analysis": Method(
        "multiple linkage analysis (MLA)",
        "which decides how many of each node's links are dominant by fitting "
        "the node's sorted link weights to ideal distributions of one, two, "
        "three, ... equally strong links, and keeps those links",
        ("vannuffel2007", "vannuffel2010", "weaver1954", "yassin2023"),
    ),
    "polya_filter": Method(
        "the Pólya filter",
        "which tests each edge's weight against a null model in which each "
        "node distributes its strength over its edges through a "
        "self-reinforcing Pólya urn process (a beta-binomial distribution "
        "with reinforcement parameter a)",
        ("marcaccioli2019",), test=True,
    ),
    "gloss_filter": Method(
        "the global statistical significance (GloSS) filter",
        "which tests each edge's weight against a global null model that keeps "
        "the topology fixed and draws every edge weight independently from the "
        "network's empirical weight distribution, conditioning on the degrees "
        "and strengths of the edge's endpoints",
        ("radicchi2011",), test=True,
    ),
    # --- structural --------------------------------------------------------
    "global_threshold_filter": Method(
        "a global weight threshold",
        "which keeps every edge whose weight is at least the threshold",
    ),
    "strongest_n_ties": Method(
        "the strongest-n-ties rule",
        "which keeps, for every node, its n heaviest edges together with "
        "any edges tied with the n-th heaviest",
    ),
    "global_sparsification": Method(
        "global sparsification",
        "which keeps the given fraction of edges with the largest weights",
        ("satuluri2011",),
    ),
    "primary_linkage_analysis": Method(
        "primary linkage analysis",
        "which keeps each node's single strongest (outgoing) edge",
        ("nystuen1961",),
    ),
    "edge_betweenness_filter": Method(
        "an edge-betweenness filter",
        "which keeps the given fraction of edges with the highest edge "
        "betweenness centrality",
        ("girvan2002",),
    ),
    "node_degree_filter": Method(
        "a node-degree filter",
        "which keeps the nodes whose degree is at least a minimum degree, "
        "together with the edges among them",
        ("freeman1978",),
    ),
    "high_salience_skeleton": Method(
        "the high-salience skeleton",
        "which scores each edge by the fraction of shortest-path trees that "
        "contain it",
        ("grady2012",),
    ),
    "metric_backbone": Method(
        "the metric backbone",
        "which keeps the edges that are shortest paths between their "
        "endpoints when path length is the sum of edge distances",
        ("simas2021",),
    ),
    "ultrametric_backbone": Method(
        "the ultrametric backbone",
        "which keeps the edges that are shortest paths between their "
        "endpoints when path length is the largest edge distance on the path",
        ("simas2021",),
    ),
    "doubly_stochastic_filter": Method(
        "the doubly stochastic filter",
        "which rescales the weight matrix to be doubly stochastic with the "
        "Sinkhorn-Knopp algorithm, scores edges by their rescaled weight, "
        "and flags the strongest edges needed to connect each component",
        ("slater2009",),
    ),
    "h_backbone": Method(
        "the h-backbone",
        "which keeps the union of the edges whose weight reaches the "
        "network's h-strength and the edges whose bridging score reaches its "
        "h-bridge",
        ("zhang2018",),
    ),
    "modularity_backbone": Method(
        "the modularity backbone",
        "which scores each node by its modularity vitality, the change in "
        "modularity when the node is removed",
        ("rajeh2022",),
    ),
    "planar_maximally_filtered_graph": Method(
        "the planar maximally filtered graph (PMFG)",
        "which adds edges from heaviest to lightest whenever the graph stays "
        "planar",
        ("tumminello2005",),
    ),
    "maximum_spanning_tree_backbone": Method(
        "the maximum spanning tree",
        "which keeps the spanning tree (forest) with the largest total weight",
    ),
    # --- proximity ---------------------------------------------------------
    "neighborhood_overlap": Method(
        "neighborhood overlap", f"which {_SHARED}", kind=U
    ),
    "jaccard_backbone": Method(
        "Jaccard similarity",
        "which scores each edge by the Jaccard similarity of its endpoints' "
        "neighborhoods",
        ("jaccard1901",), U,
    ),
    "dice_backbone": Method(
        "Dice similarity",
        "which scores each edge by the Dice similarity of its endpoints' "
        "neighborhoods",
        ("dice1945",), U,
    ),
    "cosine_backbone": Method(
        "cosine (Salton) similarity",
        "which scores each edge by the cosine similarity of its endpoints' "
        "neighborhoods",
        ("salton1983",), U,
    ),
    "hub_promoted_index": Method(
        "the hub promoted index",
        f"which {_overlap('the smaller endpoint degree')}",
        ("ravasz2002",), U,
    ),
    "hub_depressed_index": Method(
        "the hub depressed index",
        f"which {_overlap('the larger endpoint degree')}",
        ("ravasz2002",), U,
    ),
    "lhn_local_index": Method(
        "the Leicht-Holme-Newman local index",
        f"which {_overlap('the product of the endpoint degrees')}",
        ("leicht2006",), U,
    ),
    "preferential_attachment_score": Method(
        "the preferential attachment index",
        "which scores each edge by the product of its endpoint degrees",
        ("barabasi1999",), U,
    ),
    "adamic_adar_index": Method(
        "the Adamic-Adar index",
        "which scores each edge by its endpoints' shared neighbors, each "
        "weighted by the inverse logarithm of its degree",
        ("adamic2003",), U,
    ),
    "resource_allocation_index": Method(
        "the resource allocation index",
        "which scores each edge by its endpoints' shared neighbors, each "
        "weighted by its inverse degree",
        ("zhou2009",), U,
    ),
    "graph_distance_proximity": Method(
        "graph-distance proximity",
        "which scores node pairs by the reciprocal of their shortest-path "
        "distance",
        ("libennowell2007",), U,
    ),
    "local_path_index": Method(
        "the local path index",
        "which scores each edge by the number of paths of length two between "
        "its endpoints plus a down-weighted count of paths of length three",
        ("lu2009",), U,
    ),
    # --- hybrid ------------------------------------------------------------
    "glab_filter": Method(
        "the globally and locally adaptive backbone (GLAB)",
        "which tests each edge's share of its endpoints' shortest paths "
        "against a null model that spreads this involvement uniformly over "
        "the endpoint's edges, with a degree-dependent exponent",
        ("zhang2014",), test=True,
    ),
    # --- bipartite ---------------------------------------------------------
    "simple_projection": Method(
        "the simple bipartite projection",
        "which weights each pair of agents by the number of artifacts they "
        "share",
        ("coscia2019",), B,
    ),
    "hyper_projection": Method(
        "the hyperbolic bipartite projection",
        "which weights each pair of agents by their shared artifacts, each "
        "divided by its degree",
        ("coscia2019", "newman2001a", "newman2001b"), B,
    ),
    "probs_projection": Method(
        "the ProbS bipartite projection",
        "which weights each pair of agents by the probability that a "
        "two-step random walk on the bipartite graph moves between them",
        ("coscia2019", "zhou2007"), B,
    ),
    "ycn_projection": Method(
        "the YCN bipartite projection",
        "which weights each pair of agents by the stationary flow of a random "
        "walk on the projection",
        ("yildirim2014", "coscia2019"), B,
    ),
    "sdsm": Method(
        "the stochastic degree sequence model (SDSM)",
        "which tests each pair of agents' co-occurrence count against a null "
        "model that preserves the bipartite graph's agent and artifact "
        "degrees in expectation (the bipartite configuration model)",
        ("neal2014",), B, test=True,
    ),
    "fdsm": Method(
        "the fixed degree sequence model (FDSM)",
        "which tests each pair of agents' co-occurrence count against "
        "bipartite graphs sampled with exactly the observed agent and "
        "artifact degrees",
        ("neal2021", "godard2022"), B, test=True,
    ),
    "fixedfill": Method(
        "the fixed fill model",
        "which tests each pair of agents' co-occurrence count against a null "
        "model that preserves only the number of edges in the bipartite graph",
        ("neal2021",), B, test=True,
    ),
    "fixedrow": Method(
        "the fixed row model",
        "which tests each pair of agents' co-occurrence count against a null "
        "model that preserves every agent's degree exactly",
        ("neal2021",), B, test=True,
    ),
    "fixedcol": Method(
        "the fixed column model",
        "which tests each pair of agents' co-occurrence count against a null "
        "model that preserves every artifact's degree exactly",
        ("neal2021",), B, test=True,
    ),
    # --- hypergraph --------------------------------------------------------
    "mdl_hypergraph_backbone": Method(
        "the minimum description length (MDL) hypergraph backbone",
        "which keeps the set of hyperedges from which the others can be "
        "encoded most cheaply, a parameter-free information-theoretic "
        "criterion",
        ("kirkley2026",), H,
    ),
    "local_mdl_hypergraph_backbone": Method(
        "the local minimum description length (MDL) hypergraph backbone",
        "which finds the MDL backbone of each node's neighborhood (the "
        "hyperedges containing it) separately and keeps a hyperedge according "
        "to how many of its nodes' neighborhood backbones contain it",
        ("kirkley2026",), H,
    ),
    "maximal_hyperedges": Method(
        "inclusion (toplex) reduction",
        "which keeps only the hyperedges not contained in another hyperedge",
        kind=H,
    ),
    "order_filter": Method(
        "an order filter",
        "which keeps the hyperedges whose order (size) is in the given range "
        "or set",
        kind=H,
    ),
    "statistically_validated_hypergraph": Method(
        "the statistically validated hypergraph (SVH)",
        "which tests each hyperedge's multiplicity against a binomial null "
        "model that preserves node activity within each order",
        ("musciotto2021",), H, test=True, builtin_mtc="bh",
    ),
    "statistically_validated_cores": Method(
        "statistically validated cores (SVC)",
        "which tests each group of nodes' co-occurrence across hyperedges "
        "against a binomial null model that preserves node activity",
        ("musciotto2021",), H, test=True, builtin_mtc="bh",
    ),
    # --- unweighted --------------------------------------------------------
    "sparsify": Method(
        "the score-normalize-filter sparsification pipeline",
        "which scores each node's edges, optionally rank-normalizes them "
        "within the node's neighborhood, and keeps each node's top-ranked "
        "edges",
        ("neal2022",), U,
    ),
    "lspar": Method(
        "local sparsification (L-Spar)",
        "which keeps, for every node u, the ceil(d_u^s) of its edges with the "
        "highest Jaccard similarity",
        ("satuluri2011",), U,
    ),
    "local_degree": Method(
        "the local degree backbone",
        "which keeps, for every node u, the edges to its floor(d_u^s) "
        "highest-degree neighbors",
        ("lindner2015", "hamann2016"), U,
    ),
}  # fmt: skip

#: Other accepted names (short aliases and dispatch-wrapper method names).
ALIASES = {
    "disparity": "disparity_filter",
    "mlf": "marginal_likelihood_filter",
    "marginal_likelihood": "marginal_likelihood_filter",
    "lans": "lans_filter",
    "global": "global_threshold_filter",
    "global_threshold": "global_threshold_filter",
    "simple": "simple_projection",
    "hyper": "hyper_projection",
    "probs": "probs_projection",
    "ycn": "ycn_projection",
    "svh": "statistically_validated_hypergraph",
    "svc": "statistically_validated_cores",
    "localdegree": "local_degree",
}

_DISPATCH = "a dispatch wrapper; describe the method it dispatches to instead"

#: Public names in the method modules that are deliberately not described.
EXCLUDED = {
    "bipartite_projection": _DISPATCH,
    "backbone_from_projection": _DISPATCH,
    "backbone_from_weighted": _DISPATCH,
    "backbone_from_unweighted": _DISPATCH,
    "backbone": _DISPATCH,
    "bicm": "a null-model building block of sdsm; describe 'sdsm' instead",
    "fastball": "a null-model sampler used by fdsm; describe 'fdsm' instead",
    "intersection_graph": "a hypergraph representation, not a backbone",
    "s_components": "a hypergraph connectivity helper, not a backbone",
    "hypergraph_compression_ratio": "a measure of the MDL backbone",
    "HypergraphBackbone": "a result class",
    "ValidatedHypergraph": "a result class",
}

# Wording and references for each multiple-testing correction.
_MTC_TEXT = {
    "bonferroni": ("a Bonferroni correction", ()),
    "holm": ("a Holm step-down correction", ("holm1979",)),
    "hochberg": ("a Hochberg step-up correction", ("hochberg1988",)),
    "bh": ("a Benjamini-Hochberg false discovery rate correction",
           ("benjamini1995",)),
    "fdr": ("a Benjamini-Hochberg false discovery rate correction",
            ("benjamini1995",)),
    "by": ("a Benjamini-Yekutieli false discovery rate correction",
           ("benjamini2001",)),
}  # fmt: skip

_STYLES = ("text", "markdown", "bibtex")


def _registry_listing():
    """RST list of the registry, appended to the module docstring for the docs."""
    lines = ["", "Methods currently in the registry:", ""]
    for name, entry in METHODS.items():
        cites = "; ".join(REFERENCES[k].cite() for k in entry.refs)
        cites = f" ({cites})" if cites else ""
        lines.append(f"- :func:`~networkx_backbone.{name}` -- {entry.label}{cites}")
    return "\n".join(lines) + "\n"


if __doc__:  # docstrings are stripped under ``python -OO``
    __doc__ += _registry_listing()


def _resolve(method):
    """Return ``(name, Method)`` for a function or a (possibly aliased) name."""
    name = getattr(method, "__name__", method)
    if not isinstance(name, str):
        raise TypeError(f"method must be a string or a function, got {method!r}")
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key in METHODS:
        return key, METHODS[key]
    if name in EXCLUDED:
        raise ValueError(f"{name!r} is {EXCLUDED[name]}.")
    raise ValueError(
        f"Unknown method {name!r}. See networkx_backbone.describe.METHODS "
        "for the described methods."
    )


def _package_version():
    try:
        return _pkg_version(PACKAGE)
    except PackageNotFoundError:  # running from an uninstalled checkout
        return "unknown"


def _package_text(version, markdown):
    title = "networkx-backbone: Backbone extraction algorithms for complex networks"
    if markdown:
        title = f"*{title}*"
    url = f"<{REPOSITORY}>" if markdown else REPOSITORY
    return f"Keegan, B. C. (n.d.). {title} (Version {version}) [Python package]. {url}"


def _package_bibtex(version):
    return (
        "@misc{keegan_networkx_backbone,\n"
        "  author = {Keegan, Brian C.},\n"
        "  title = {networkx-backbone: Backbone extraction algorithms for "
        "complex networks},\n"
        f"  version = {{{version}}},\n"
        f"  url = {{{REPOSITORY}}},\n"
        "  note = {Python package}\n"
        "}"
    )


def _pct(x):
    return f"{100 * x:.1f}%"


def _fmt_param(name, value, markdown):
    return f"`{name}={value!r}`" if markdown else f"{name}={value!r}"


def _hyperedges(obj):
    """Distinct hyperedges of a hyperedge collection or a result object."""
    for attr in ("backbone", "validated"):
        if hasattr(obj, attr) and not isinstance(obj, nx.Graph):
            obj = getattr(obj, attr)
            break
    return {frozenset(e) for e in obj}


def _graph_share(original, backbone, weight):
    from networkx_backbone.measures import (
        edge_fraction,
        node_fraction,
        weight_fraction,
    )

    m, m_bb = original.number_of_edges(), backbone.number_of_edges()
    n = sum(1 for v in original if original.degree(v) > 0)
    n_bb = sum(1 for v in backbone if backbone.degree(v) > 0)
    text = (
        f"The backbone retains {m_bb} of {m} edges "
        f"({_pct(edge_fraction(original, backbone))}) and {n_bb} of {n} "
        f"non-isolated nodes ({_pct(node_fraction(original, backbone))})"
    )
    weights = [d.get(weight) for _, _, d in original.edges(data=True)]
    if weights and all(
        isinstance(w, int | float) and not isinstance(w, bool) for w in weights
    ):
        share = weight_fraction(original, backbone, weight=weight)
        text += f", and {_pct(share)} of the total edge weight"
    return text + "."


def _size_sentence(entry, original, backbone, weight):
    if backbone is None:
        return ""
    if entry.kind == H:
        kept = _hyperedges(backbone)
        if original is None:
            return f"The backbone contains {len(kept)} hyperedges."
        total = _hyperedges(original)
        share = len(kept) / len(total) if total else 0.0
        return (
            f"The backbone retains {len(kept)} of {len(total)} distinct "
            f"hyperedges ({_pct(share)})."
        )
    if not isinstance(backbone, nx.Graph):
        raise TypeError("backbone must be a NetworkX graph for this method")
    if entry.kind == B or original is None:
        n, m = backbone.number_of_nodes(), backbone.number_of_edges()
        what = "agents" if entry.kind == B else "nodes"
        text = f"The backbone contains {m} edges among {n} {what}"
        if n > 1 and entry.kind == B:
            text += f" (density {nx.density(backbone):.3f})"
        return text + "."
    if not isinstance(original, nx.Graph):
        raise TypeError("original must be a NetworkX graph for this method")
    return _graph_share(original, backbone, weight)


def _num(x):
    return f"{x:g}" if isinstance(x, int | float) else repr(x)


#: Scorers whose ``unit=`` converts weights to counts.
_UNIT_METHODS = frozenset(
    {"marginal_likelihood_filter", "ecm_filter", "noise_corrected_filter"}
)


def _model_sentences(name, params, backbone, md):
    """Sentences for model choices that change the null (``a``, ``unit``).

    Returns ``(sentences, reference_keys, remaining_params)``; the parameters
    described here are removed from the generic "Other parameters" list.
    """
    params = dict(params)
    sentences, keys = [], []

    def code(text):
        return f"`{text}`" if md else text

    if name == "disparity_filter" and "a" in params:
        a = params.pop("a")
        if a == "ml":
            fitted = getattr(backbone, "graph", {}).get("disparity_a")
            value = (
                f"estimated by maximum likelihood ({code(f'a = {fitted:.3g}')})"
                if isinstance(fitted, float)
                else "estimated by maximum likelihood"
            )
        else:
            value = code(f"a = {_num(a)}")
        if a != 1:
            keys.append("marcaccioli2019")
            sentences.append(
                "The uniform null model was generalized to a symmetric "
                "Dirichlet distribution of each node's weight shares with "
                f"concentration 1/a, {value}; this is the large-strength limit "
                "of the Pólya urn null model "
                f"({REFERENCES['marcaccioli2019'].cite()})."
            )
    if name in _UNIT_METHODS and "unit" in params:
        unit = params.pop("unit")
        if unit != 1:
            sentences.append(
                "Edge weights were converted to counts in units of "
                f"{code(f'unit = {_num(unit)}')} (count = weight / unit) before "
                "testing."
            )
    return sentences, keys, params


def _transform_sentences(original, backbone, md):
    """Sentences for weight transforms recorded by ``transform_weights``."""
    records = []
    for graph in (backbone, original):
        if isinstance(graph, nx.Graph) and graph.graph.get("weight_transforms"):
            records = graph.graph["weight_transforms"]
            break
    sentences = []
    for rec in records:
        how = f"`{rec['transform']}`" if md else rec["transform"]
        src, dst = repr(rec["weight"]), repr(rec["out"])
        into = "" if src == dst else f" into the attribute {dst}"
        sentences.append(
            f"Before scoring, the edge weights ({src}) were transformed with "
            f"{how}{into}, so the null model applies to the transformed "
            "weights."
        )
    return sentences


def _sign_sentence(backbone):
    if not isinstance(backbone, nx.Graph):
        return ""
    signs = [d.get("sign") for _, _, d in backbone.edges(data=True)]
    if not signs or any(s is None for s in signs):
        return ""
    pos = sum(1 for s in signs if s > 0)
    return (
        f" Of the retained edges, {pos} are significantly strong (sign +1) "
        f"and {len(signs) - pos} significantly weak (sign -1)."
    )


def describe_backbone(
    method: str | Callable[..., Any],
    *,
    alpha: float | None = None,
    mtc: str = "none",
    signed: bool = False,
    original: nx.Graph | Hyperedges | None = None,
    backbone: nx.Graph
    | Hyperedges
    | HypergraphBackbone
    | ValidatedHypergraph
    | None = None,
    style: str = "text",
    **params: Any,
) -> str:
    """Write a methods paragraph and citations for a backbone.

    The counterpart of the ``narrative`` option of Neal's *Backbone 3.0*:
    the returned text names the method and its null model (or criterion),
    lists the parameters, states the multiple-testing correction, reports
    how much of the network the backbone keeps (via
    :mod:`~networkx_backbone.measures`) and cites the method's papers and
    this package, with its installed version.

    Parameters
    ----------
    method : string or callable
        Name of a backbone function such as ``"disparity_filter"``, a short
        alias such as ``"disparity"`` or ``"svh"``, a method name accepted
        by :func:`~networkx_backbone.backbone` such as ``"global"``, or the
        function itself.  Every described method is listed in
        ``networkx_backbone.describe.METHODS``.
    alpha : float or None, optional (default=None)
        Significance level the p-values were compared with.  Only valid for
        methods that produce p-values.
    mtc : string, optional (default="none")
        Multiple-testing correction applied before thresholding, as passed
        to :func:`~networkx_backbone.threshold_filter` (see
        :func:`~networkx_backbone.adjust_pvalues`).
    signed : bool, optional (default=False)
        Whether a two-tailed (signed) test was run.  Only valid for methods
        that take a ``signed`` argument.
    original : graph or iterable of hyperedges, optional
        The input network.  With *backbone*, the share of edges, nodes and
        weight kept is reported.
    backbone : graph, iterable of hyperedges or result object, optional
        The extracted backbone.
    style : {"text", "markdown", "bibtex"}, optional (default="text")
        ``"text"`` and ``"markdown"`` return the paragraph followed by a
        reference list; ``"bibtex"`` returns only the BibTeX entries.
    **params
        Other parameters used (for example ``s=0.5`` or ``weight="weight"``);
        they are listed in the paragraph.  A ``weight`` entry also names the
        edge attribute used for the weight share.  Parameters that change the
        null model get a sentence of their own: ``a`` of
        :func:`~networkx_backbone.disparity_filter` (with ``a="ml"`` the
        estimate is read from ``backbone.graph["disparity_a"]``) and ``unit``
        of the count-based scorers.  Weight transforms recorded by
        :func:`~networkx_backbone.transform_weights` in the graph attributes
        of *backbone* (or else *original*) are reported too.

    Returns
    -------
    description : string
        The paragraph and reference list, or BibTeX entries.

    Raises
    ------
    ValueError
        If *method* is not described in the registry, *style* or *mtc* is
        unknown, or *alpha*, *mtc* or *signed* is given for a method that
        does not support it.
    TypeError
        If *original* or *backbone* has the wrong type for the method.

    See Also
    --------
    threshold_filter : Apply *alpha* and *mtc* to a scored graph.
    compare_backbones : Compare several backbones on the same measures.

    Notes
    -----
    The package is cited without a year ("n.d.") because the version, not
    the date, identifies the code; ``CITATION.cff`` in the repository
    carries the same metadata.

    Examples
    --------
    A disparity-filter backbone of Les Misérables at ``alpha = 0.05``:

    >>> import networkx as nx
    >>> import networkx_backbone as nb
    >>> G = nx.les_miserables_graph()
    >>> scored = nb.disparity_filter(G)
    >>> bb = nb.threshold_filter(scored, "disparity_pvalue", 0.05)
    >>> text = nb.describe_backbone(
    ...     "disparity_filter", alpha=0.05, original=G, backbone=bb
    ... )
    >>> print(text)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    We extracted the backbone with the disparity filter (Serrano et al.,
    2009), which tests each edge's weight against a null model in which
    each node's strength is spread uniformly at random over its edges.
    Edges were kept if their p-value was below alpha = 0.05, without
    correction for multiple comparisons. The backbone retains 9 of 254
    edges (3.5%) and 10 of 77 non-isolated nodes (13.0%), and 17.1% of the
    total edge weight. Backbones were computed with networkx-backbone
    version ... (Keegan, n.d.).
    <BLANKLINE>
    References
    <BLANKLINE>
    Keegan, B. C. (n.d.). networkx-backbone: Backbone extraction algorithms
    for complex networks (Version ...) [Python package].
    https://github.com/brianckeegan/networkx_backbone
    Serrano, M. A., Boguñá, M., & Vespignani, A. (2009). Extracting the
    multiscale backbone of complex weighted networks. Proceedings of the
    National Academy of Sciences, 106(16), 6483-6488.
    https://doi.org/10.1073/pnas.0808904106

    The BibTeX entries for an L-Spar backbone:

    >>> print(nb.describe_backbone("lspar", s=0.5, style="bibtex"))  # doctest: +ELLIPSIS
    @inproceedings{satuluri2011,
      author = {Satuluri, V. and Parthasarathy, S. and Ruan, Y.},
      title = {Local graph sparsification for scalable clustering},
      booktitle = {Proceedings of the ACM SIGMOD International Conference on Management of Data},
      pages = {721--732},
      year = {2011},
      doi = {10.1145/1989323.1989399}
    }
    <BLANKLINE>
    @misc{keegan_networkx_backbone,
      author = {Keegan, Brian C.},
      title = {networkx-backbone: Backbone extraction algorithms for complex networks},
      version = {...},
      url = {https://github.com/brianckeegan/networkx_backbone},
      note = {Python package}
    }
    """
    name, entry = _resolve(method)
    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}, got {style!r}")

    from networkx_backbone.filters import _MTC_METHODS

    mtc_l = str(mtc).lower()
    if mtc_l not in _MTC_METHODS:
        raise ValueError(f"mtc must be one of {_MTC_METHODS}, got {mtc!r}")
    if not entry.test and (alpha is not None or mtc_l != "none"):
        raise ValueError(
            f"{name} does not produce p-values, so alpha and mtc do not apply"
        )
    if entry.builtin_mtc and mtc_l != "none":
        raise ValueError(
            f"{name} applies its own {entry.builtin_mtc!r} correction; leave mtc='none'"
        )
    if signed:
        import networkx_backbone as nb

        if "signed" not in inspect.signature(getattr(nb, name)).parameters:
            raise ValueError(f"{name} has no signed (two-tailed) variant")

    keys = list(entry.refs)
    if name == "sdsm" and ({"prohibited", "required"} & params.keys()):
        keys.append("neal2023")
    method_cites = "; ".join(REFERENCES[k].cite() for k in keys)
    model_sentences, model_keys, params = _model_sentences(
        name, params, backbone, style == "markdown"
    )
    keys += [k for k in model_keys if k not in keys]

    if mtc_l != "none":
        mtc_phrase, mtc_keys = _MTC_TEXT[mtc_l]
    elif entry.builtin_mtc:
        mtc_phrase, mtc_keys = _MTC_TEXT[entry.builtin_mtc]
    else:
        mtc_phrase, mtc_keys = None, ()
    keys += [k for k in mtc_keys if k not in keys]

    version = _package_version()
    if style == "bibtex":
        entries = [REFERENCES[k].bibtex() for k in keys]
        return "\n\n".join([*entries, _package_bibtex(version)])

    md = style == "markdown"
    inner = "; ".join(filter(None, [f"`{name}`" if md else "", method_cites]))
    cite = f" ({inner})" if inner else ""
    verb = (
        "We projected the bipartite network with"
        if entry.kind == B and not entry.test
        else "We extracted the backbone with"
    )
    sentences = [f"{verb} {entry.label}{cite}, {entry.summary}."]
    sentences += model_sentences
    sentences += _transform_sentences(original, backbone, md)

    if signed:
        sentences.append(
            "A two-tailed test was used, so the signed backbone keeps both "
            "significantly strong and significantly weak edges, marked by a "
            "sign attribute of +1 and -1."
        )
    if entry.test:
        unit = {H: "Hyperedges", B: "Agent pairs"}.get(entry.kind, "Edges")
        mtc_cite = "; ".join(REFERENCES[k].cite() for k in mtc_keys)
        mtc_cite = f" ({mtc_cite})" if mtc_cite else ""
        if alpha is None:
            sentence = "P-values were computed for every candidate"
            if mtc_phrase:
                sentence += f" and adjusted with {mtc_phrase}{mtc_cite}"
        else:
            a = f"`alpha = {alpha:g}`" if md else f"alpha = {alpha:g}"
            sentence = f"{unit} were kept if their p-value was below {a}"
            if mtc_phrase and entry.builtin_mtc:
                sentence += f", with {mtc_phrase} applied within the method{mtc_cite}"
            elif mtc_phrase:
                sentence += f" after {mtc_phrase} for multiple comparisons{mtc_cite}"
            else:
                sentence += ", without correction for multiple comparisons"
        sentences.append(sentence + ".")
    if params:
        listed = ", ".join(_fmt_param(k, v, md) for k, v in params.items())
        sentences.append(f"Other parameters: {listed}.")

    size = _size_sentence(entry, original, backbone, params.get("weight", "weight"))
    if size:
        if signed:
            size += _sign_sentence(backbone)
        sentences.append(size)
    sentences.append(
        f"Backbones were computed with networkx-backbone version {version} "
        "(Keegan, n.d.)."
    )

    refs = [REFERENCES[k] for k in keys]
    entries = [(r.authors[0], r.year, r.text(markdown=md)) for r in refs]
    entries.append(("Keegan, B. C.", 0, _package_text(version, md)))
    entries.sort(key=lambda t: (t[0].lower(), t[1]))
    if md:
        reflist = "\n".join(f"- {t}" for *_, t in entries)
        return " ".join(sentences) + "\n\n**References**\n\n" + reflist
    return (
        " ".join(sentences) + "\n\nReferences\n\n" + "\n".join(t for *_, t in entries)
    )
