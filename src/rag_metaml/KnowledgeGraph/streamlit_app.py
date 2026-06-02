from __future__ import annotations

import os
import sys
import tempfile

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ── path setup so we can import from rag_metaml ──────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from rag_metaml.KnowledgeGraph.graph import build_taxonomy_graph
from rag_metaml.hybrid_search import hybrid_search

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MetaML Knowledge Graph",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── load graph once ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading taxonomy graph…")
def load_graph() -> nx.DiGraph:
    return build_taxonomy_graph()


graph = load_graph()

# ── helpers ───────────────────────────────────────────────────────────────────
NODE_COLORS = {
    "domain": "#e74c3c",      # red
    "dimension": "#3498db",   # blue
    "area": "#2ecc71",        # green
}
NODE_SIZES = {
    "domain": 40,
    "dimension": 25,
    "area": 12,
}


def _domain_subtree(graph: nx.DiGraph, domain_node: str) -> set[str]:
    """Return all descendant node IDs of a domain node."""
    return set(nx.descendants(graph, domain_node)) | {domain_node}


def build_pyvis(
    graph: nx.DiGraph,
    selected_domain: str | None = None,
    show_areas: bool = False,
    highlight: set[str] | None = None,
) -> str:
    """Build a pyvis HTML string from the networkx graph."""
    net = Network(
        height="680px",
        width="100%",
        bgcolor="#0f1117",
        font_color="#ffffff",
        directed=True,
    )
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
        "color": { "color": "#555577", "highlight": "#aaaaff" },
        "smooth": { "type": "dynamic" }
      }
    }
    """)

    # Determine which nodes to include
    if selected_domain:
        allowed = _domain_subtree(graph, selected_domain)
    else:
        # all nodes, but optionally without areas
        allowed = set(graph.nodes())

    if not show_areas:
        allowed = {n for n in allowed if graph.nodes[n].get("label") != "area"}

    highlight = highlight or set()

    for node_id in allowed:
        attrs = graph.nodes[node_id]
        label_type = attrs.get("label", "area")
        name = attrs.get("name") or node_id
        color = "#f39c12" if node_id in highlight else NODE_COLORS.get(label_type, "#95a5a6")
        border = "#ffffff" if node_id in highlight else color
        size = NODE_SIZES.get(label_type, 12)

        net.add_node(
            node_id,
            label=name,
            title=f"<b>{name}</b><br>Type: {label_type}<br>ID: {node_id}",
            color={"background": color, "border": border, "highlight": {"background": "#f39c12"}},
            size=size,
            font={"size": 11 if label_type == "area" else 14},
            shape="dot",
        )

    for src, dst, edata in graph.edges(data=True):
        if src in allowed and dst in allowed:
            relation = edata.get("relation", "→")
            net.add_edge(src, dst, title=relation, width=1.5)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as tmp:
        path = tmp.name

    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(path)
    return html


def format_rag_response(results: dict) -> str:
    """Turn hybrid_search output into a readable markdown response."""
    query = results.get("query", "")
    seeds = results.get("seeds", [])
    candidates = results.get("candidates", [])
    counts = results.get("counts", {})

    lines = [
        f"I searched the MetaML taxonomy for **\"{query}\"** and found "
        f"**{counts.get('seed_count', 0)} direct matches**, expanding to "
        f"**{counts.get('candidate_count', 0)} related concepts** via graph traversal.\n"
    ]

    if seeds:
        lines.append("### Top semantic matches")
        for s in seeds[:5]:
            dist = s.get("distance", 0)
            score = round((1 - dist) * 100, 1)
            lines.append(
                f"- **{s['name']}** `{s['label']}`  —  similarity {score}%"
            )

    if candidates:
        lines.append("\n### Full context (graph-expanded)")
        grouped: dict[str, list] = {"domain": [], "dimension": [], "area": []}
        for c in candidates:
            grouped.setdefault(c.get("label", "area"), []).append(c)

        for tier in ("domain", "dimension", "area"):
            items = grouped.get(tier, [])
            if not items:
                continue
            tier_label = tier.capitalize() + "s"
            lines.append(f"\n**{tier_label}:**")
            for c in items[:12]:
                path = c.get("path") or []
                path_str = " › ".join(path) if path else c["name"]
                lines.append(f"  - {path_str}")
            if len(items) > 12:
                lines.append(f"  - _…and {len(items) - 12} more_")

    return "\n".join(lines)


# ── sidebar legend ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## MetaML Knowledge Graph")
    st.markdown(
        """
        **Node types**
        - 🔴 **Domain** — top-level sector (e.g. FinTech)
        - 🔵 **Dimension** — sub-category (e.g. Banking)
        - 🟢 **Area** — leaf concept (e.g. Mobile Banking)
        - 🟠 **Highlighted** — search match
        """
    )
    st.divider()
    st.caption(
        f"Graph: {graph.number_of_nodes()} nodes · {graph.number_of_edges()} edges"
    )

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_graph, tab_chat = st.tabs(["🕸️  Graph Explorer", "💬  RAG Chatbot"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GRAPH EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.markdown("## Knowledge Graph Explorer")
    st.caption(
        "Use the controls below to filter the graph. "
        "Zoom, pan and hover over nodes for details."
    )

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 3])

    with ctrl_col1:
        domain_nodes = sorted(
            [(n, d["name"]) for n, d in graph.nodes(data=True) if d.get("label") == "domain"],
            key=lambda x: x[1],
        )
        domain_options = {"All domains": None, **{name: nid for nid, name in domain_nodes}}
        selected_label = st.selectbox("Domain filter", list(domain_options.keys()))
        selected_domain_id = domain_options[selected_label]

    with ctrl_col2:
        show_areas = st.checkbox("Show Areas (622 nodes)", value=False)

    with ctrl_col3:
        search_query = st.text_input("🔍 Highlight nodes by name", placeholder="e.g. blockchain")

    # Find highlight set
    highlight_ids: set[str] = set()
    if search_query:
        q = search_query.lower()
        for node_id, attrs in graph.nodes(data=True):
            name = (attrs.get("name") or "").lower()
            if q in name:
                highlight_ids.add(node_id)
        if highlight_ids:
            st.success(f"Found {len(highlight_ids)} node(s) matching '{search_query}' — shown in orange")
        else:
            st.warning(f"No nodes matched '{search_query}'")

    # Render graph
    with st.spinner("Rendering graph…"):
        html = build_pyvis(graph, selected_domain_id, show_areas, highlight_ids)

    components.html(html, height=700, scrolling=False)

    # Node info table for search results
    if highlight_ids:
        st.markdown("#### Matching nodes")
        rows = []
        for nid in sorted(highlight_ids):
            attrs = graph.nodes[nid]
            path = []
            for pred in nx.ancestors(graph, nid):
                path.append(graph.nodes[pred].get("name") or pred)
            rows.append({
                "ID": nid,
                "Name": attrs.get("name") or nid,
                "Type": attrs.get("label"),
                "Parents": ", ".join(
                    graph.nodes[p].get("name") or p
                    for p in graph.predecessors(nid)
                ),
            })
        st.dataframe(rows, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## MetaML RAG Assistant")
    st.caption(
        "Ask anything about the taxonomy. "
        "The assistant uses semantic search + graph expansion to find answers."
    )

    # init session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your MetaML taxonomy assistant. "
                    "Ask me about any concept — e.g. *'What is WealthTech?'*, "
                    "*'Show me areas under Payments'*, or *'Tell me about blockchain in FinTech'*."
                ),
            }
        ]

    # chat controls
    chat_col, settings_col = st.columns([3, 1])

    with settings_col:
        st.markdown("**Search settings**")
        top_k = st.slider("Top K seeds", 1, 10, 5)
        expand_hops = st.slider("Graph hops", 0, 3, 2)
        direction = st.radio("Expand direction", ["both", "down", "up"], index=0)
        label_filter = st.selectbox(
            "Filter by node type", ["(any)", "domain", "dimension", "area"]
        )
        label_filter_val = None if label_filter == "(any)" else label_filter

        if st.button("🗑️ Clear chat"):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()

    with chat_col:
        # render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # input
        if user_input := st.chat_input("Ask about the MetaML taxonomy…"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching taxonomy…"):
                    try:
                        results = hybrid_search(
                            graph,
                            user_input,
                            top_k=top_k,
                            expand_hops=expand_hops,
                            direction=direction,
                            label_filter=label_filter_val,
                        )
                        response = format_rag_response(results)
                    except Exception as exc:
                        response = f"⚠️ Search failed: {exc}"

                st.markdown(response)

                # show raw seeds in expander
                if "seeds" in results:
                    with st.expander("🔬 Debug: raw seed matches"):
                        for s in results["seeds"]:
                            st.json(s)

            st.session_state.messages.append({"role": "assistant", "content": response})
