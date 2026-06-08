"""
frontend/app.py
---------------
Streamlit frontend for MetaML Taxonomy — UC1 / UC2 / UC4.

Run from RAG_METAML/ project root:
  streamlit run frontend/app.py

API must be running first:
  uvicorn index.api:app --reload --port 8000
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


def _resolve_api_base() -> str:
    """Local dev → localhost; Streamlit Cloud → Vercel API (or secrets override)."""
    if os.getenv("API_BASE_URL"):
        return os.getenv("API_BASE_URL").rstrip("/")
    try:
        return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except Exception:
        pass
    return "https://ragmetaml.vercel.app"


API_BASE = _resolve_api_base()

st.set_page_config(
    page_title="MetaML Taxonomy",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea { font-family: 'Syne', sans-serif; font-size: 14px; border-radius: 8px; }
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
.path-crumb { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #6b7280; }
.node-pill { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:11px; padding:2px 8px; border-radius:20px; margin-right:4px; margin-bottom:4px; }
.pill-domain  { background:#eeedfe; color:#3c3489; }
.pill-dim     { background:#e1f5ee; color:#085041; }
.pill-area    { background:#faeeda; color:#633806; }
.pill-info    { background:#f1f0e8; color:#444441; }
.tree-line { font-family:'IBM Plex Mono',monospace; font-size:12px; line-height:1.7; white-space:pre; }
h1 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 500 !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ────────────────────────────────────────────────────────────────
def api_get(path: str, params: dict = None) -> Optional[Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API. Make sure it is running:\n\n"
            "```\nuvicorn index.api:app --reload --port 8000\n```"
        )
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, body: dict) -> Optional[Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API. Make sure it is running:\n\n"
            "```\nuvicorn index.api:app --reload --port 8000\n```"
        )
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Shared components ──────────────────────────────────────────────────────────
def render_path_crumb(path_str: str):
    parts = path_str.split(" > ")
    colors = ["#534ab7", "#0f6e56", "#854f0b"]
    html = ""
    for i, p in enumerate(parts):
        c = colors[min(i, len(colors) - 1)]
        html += f'<span style="color:{c};font-family:IBM Plex Mono,monospace;font-size:12px">{p}</span>'
        if i < len(parts) - 1:
            html += '<span style="color:#b4b2a9;font-family:IBM Plex Mono,monospace;font-size:12px"> › </span>'
    st.markdown(html, unsafe_allow_html=True)


def render_score_bar(score: float, label: str = ""):
    pct = int(score * 100)
    color = "#1d9e75" if pct >= 70 else "#ba7517" if pct >= 45 else "#d85a30"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
        f'<div style="flex:1;background:#f1f0e8;border-radius:4px;height:5px">'
        f'<div style="width:{pct}%;background:{color};border-radius:4px;height:5px"></div></div>'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:12px;color:{color};min-width:36px">{pct}%</span>'
        f'{"<span style=font-size:12px;color:#888780>" + label + "</span>" if label else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def label_pill(label: str, text: str = None):
    cls = {"domain": "pill-domain", "dimension": "pill-dim", "area": "pill-area"}.get(label, "pill-info")
    st.markdown(f'<span class="node-pill {cls}">{text or label}</span>', unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_health = st.columns([5, 1])
with col_title:
    st.markdown("## ⬡ MetaML Taxonomy")
with col_health:
    if st.button("Check API", use_container_width=True):
        h = api_get("/health")
        if h:
            g, e = h.get("graph", {}), h.get("embedder", {})
            st.success(f"Graph: {g.get('total_nodes',0)} nodes | Index: {e.get('indexed_nodes',0)} areas")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "UC1 · Semantic Search",
    "UC2 · Graph Explorer",
    "UC3 · Similarity Mapper",
    "UC4 · Classifier",
    "UC5 · Coverage Audit",
])


# ══════════════════════════════════════════════════════════════════════
# UC1 — SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Find taxonomy areas by meaning, not just keywords")
    st.caption("Query is embedded with a sentence transformer and matched against full path strings (Domain > Dimension > Area).")

    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Search query", label_visibility="collapsed",
            placeholder='"stable coins", "remote patient monitoring", "crypto lending"',
        )
    with col_k:
        top_k = st.selectbox("Results", [3, 5, 10], index=1, label_visibility="collapsed")

    if st.button("Search", type="primary", disabled=not query):
        with st.spinner("Encoding query and searching..."):
            results = api_post("/search", {"query": query, "top_k": top_k})

        if results is None:
            st.stop()
        elif not results:
            st.warning("No results. Try a different query.")
        else:
            st.markdown(f"**{len(results)} results** for *{query}*")
            st.divider()
            for i, res in enumerate(results):
                c1, c2 = st.columns([1, 11])
                with c1:
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:24px;'
                        f'color:#d3d1c7;font-weight:500;padding-top:4px">#{i+1}</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    top, score_col = st.columns([3, 1])
                    with top:
                        st.markdown(f"**{res['area_name']}**")
                        render_path_crumb(res["path"])
                    with score_col:
                        render_score_bar(res["score"], "similarity")

                    siblings = res.get("siblings", [])
                    if siblings:
                        st.markdown('<span style="font-size:11px;color:#888780">Related in same dimension:</span>', unsafe_allow_html=True)
                        pills = "".join(f'<span class="node-pill pill-area">{s["name"]}</span>' for s in siblings[:6])
                        st.markdown(pills, unsafe_allow_html=True)
                st.divider()


# ══════════════════════════════════════════════════════════════════════
# UC2 — GRAPH EXPLORER
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Explore the taxonomy graph by node name")
    st.caption("Find any domain, dimension, or area. Inspect parents, children, and the full successor subtree.")

    col_input, col_depth, col_dir = st.columns([4, 1, 1])
    with col_input:
        node_query = st.text_input(
            "Node name", label_visibility="collapsed", key="uc2_query",
            placeholder='"Banking", "FinTech", "Stablecoins"',
        )
    with col_depth:
        depth = st.selectbox("Depth", [1, 2, 3, 4], index=0, label_visibility="collapsed")
    with col_dir:
        direction = st.selectbox("Direction", ["both", "up", "down"], label_visibility="collapsed")

    if st.button("Find node", type="primary", disabled=not node_query):
        name_hits = api_get("/search-name", params={"q": node_query, "limit": 15})
        if name_hits:
            st.session_state["uc2_hits"] = name_hits
            st.session_state["uc2_selected"] = None

    if st.session_state.get("uc2_hits"):
        hits = st.session_state["uc2_hits"]
        options = {
            f"{h['name']}  [{h['label']}]  —  {h['path_str']}": h["node_id"]
            for h in hits
        }
        chosen_label = st.selectbox(f"Found {len(hits)} match(es) — pick one:", list(options.keys()))
        selected_id = options[chosen_label]

        if st.button("Explore", type="secondary") or st.session_state.get("uc2_selected") == selected_id:
            st.session_state["uc2_selected"] = selected_id

            node_detail = api_get(f"/node/{selected_id}")
            neighbors   = api_get(f"/neighbors/{selected_id}", {"depth": depth, "direction": direction})
            subtree     = api_get(f"/subtree/{selected_id}", {"max_nodes": 150})

            if node_detail is None:
                st.stop()

            st.divider()
            hc, bc = st.columns([5, 1])
            with hc:
                st.markdown(f"### {node_detail['name']}")
                render_path_crumb(node_detail["path_str"])
            with bc:
                label_pill(node_detail["label"], node_detail["label"].upper())
            st.caption(f"node id: `{node_detail['node_id']}`")

            st.markdown(f"**Neighbors** (depth={depth}, direction={direction})")
            if neighbors:
                by_label: Dict[str, list] = {}
                for n in neighbors:
                    by_label.setdefault(n["label"], []).append(n)
                for lbl, nodes in by_label.items():
                    cls = {"domain":"pill-domain","dimension":"pill-dim","area":"pill-area"}.get(lbl,"pill-info")
                    pills = "".join(
                        f'<span class="node-pill {cls}" title="{n["path_str"]}">{n["name"]}</span>'
                        for n in nodes
                    )
                    st.markdown(
                        f'<span style="font-size:11px;color:#888780;font-family:IBM Plex Mono,monospace">{lbl}</span><br>{pills}',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No neighbors at this depth/direction.")

            st.divider()
            st.markdown("**Successor tree** (from this node to leaves)")
            if subtree:
                tree_html = f'<div class="tree-line">{node_detail["name"]}\n'
                for item in subtree:
                    branch = "└─ " if item["is_last"] else "├─ "
                    lbl_color = {"domain":"#534ab7","dimension":"#0f6e56","area":"#854f0b"}.get(item["label"],"#888780")
                    tree_html += (
                        item["prefix"] + branch
                        + f'<span style="color:{lbl_color}">{item["name"]}</span>'
                        + f'<span style="color:#d3d1c7;font-size:10px">  [{item["label"]}]</span>\n'
                    )
                tree_html += "</div>"
                st.markdown(tree_html, unsafe_allow_html=True)
            else:
                st.caption("This node has no successors (leaf node).")


# ══════════════════════════════════════════════════════════════════════
# UC4 — CLASSIFIER
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Classify a company or product description into taxonomy paths")
    st.caption("Paste free-form text. Embeddings find the closest taxonomy areas; graph walks up to return full path + confidence.")

    example_texts = {
        "Fintech startup": (
            "We offer a peer-to-peer lending marketplace for small businesses, "
            "connecting borrowers directly with institutional investors. "
            "Our platform uses AI for credit scoring and risk assessment."
        ),
        "HealthTech product": (
            "Our wearable device continuously monitors blood glucose levels "
            "and sends real-time data to a mobile app, enabling diabetic patients "
            "to manage their condition without finger-prick tests."
        ),
        "Crypto platform": (
            "A decentralized exchange protocol built on Ethereum that allows users "
            "to swap ERC-20 tokens using automated market makers and liquidity pools. "
            "We also offer staking rewards for liquidity providers."
        ),
    }

    ex_col, _ = st.columns([3, 5])
    with ex_col:
        chosen_example = st.selectbox("Load an example", ["(none)"] + list(example_texts.keys()))

    input_text = st.text_area(
        "Text to classify",
        value=example_texts.get(chosen_example, "") if chosen_example != "(none)" else "",
        height=140,
        placeholder="Paste a company description, product blurb, or any text...",
        label_visibility="collapsed",
    )

    n_results = st.radio("Paths to return", [1, 2, 3], index=2, horizontal=True)

    if st.button("Classify", type="primary", disabled=not input_text.strip()):
        with st.spinner("Embedding and classifying..."):
            results = api_post("/classify", {"text": input_text, "top_k": n_results})

        if results is None:
            st.stop()
        elif not results:
            st.warning("No results. Try more descriptive text.")
        else:
            st.divider()
            st.markdown(f"**Top {len(results)} classification(s)**")

            for i, res in enumerate(results):
                rank_label = {0: "Primary", 1: "Secondary", 2: "Tertiary"}.get(i, f"#{i+1}")
                with st.expander(
                    f"{rank_label}: {res['path']}  ·  {res['confidence_pct']}% confidence",
                    expanded=(i == 0),
                ):
                    conf_col, path_col = st.columns([1, 3])
                    with conf_col:
                        st.markdown(
                            f'<div style="text-align:center">'
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:32px;font-weight:500;color:#1d9e75">'
                            f'{res["confidence_pct"]}%</div>'
                            f'<div style="font-size:11px;color:#888780">confidence</div></div>',
                            unsafe_allow_html=True,
                        )
                        render_score_bar(res["score"])
                    with path_col:
                        st.markdown("**Taxonomy path**")
                        render_path_crumb(res["path"])
                        st.markdown(f"**Area:** {res['area_name']}")
                        parent_ctx = res.get("parent_context", [])
                        if parent_ctx:
                            st.markdown("**Graph ancestors:**")
                            pills = "".join(
                                '<span class="node-pill {}">{}</span>'.format(
                                    {"domain":"pill-domain","dimension":"pill-dim"}.get(p["label"],"pill-info"),
                                    p["name"],
                                )
                                for p in parent_ctx
                            )
                            st.markdown(pills, unsafe_allow_html=True)

            st.divider()
            export = {
                "input_text": input_text[:200] + ("..." if len(input_text) > 200 else ""),
                "classifications": [
                    {"rank": i+1, "path": r["path"], "area": r["area_name"],
                     "confidence_pct": r["confidence_pct"], "score": r["score"], "node_id": r["node_id"]}
                    for i, r in enumerate(results)
                ],
            }
            st.download_button(
                "Export as JSON", data=json.dumps(export, indent=2),
                file_name="taxonomy_classification.json", mime="application/json",
            )


# ══════════════════════════════════════════════════════════════════════
# UC3 — CONCEPT SIMILARITY MAPPER
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Compare two taxonomy concepts by semantic + structural distance")
    st.caption(
        "Pick any two area nodes. Get their cosine similarity from embeddings "
        "and the shortest path between them through the knowledge graph."
    )

    # ── Node pickers (reuse /search-name) ─────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Concept A**")
        q_a = st.text_input("Search concept A", placeholder='"Stablecoins", "Robo-Advisors"', key="uc3_qa")
        if st.button("Find A", key="uc3_find_a", disabled=not q_a):
            hits = api_get("/search-name", {"q": q_a, "limit": 10})
            if hits:
                st.session_state["uc3_hits_a"] = hits

        if st.session_state.get("uc3_hits_a"):
            opts_a = {f"{h['name']}  [{h['label']}]": h["node_id"] for h in st.session_state["uc3_hits_a"]}
            chosen_a = st.selectbox("Select A", list(opts_a.keys()), key="uc3_sel_a")
            st.session_state["uc3_node_a"] = opts_a[chosen_a]
            st.caption(f"`{st.session_state['uc3_node_a']}`")

    with col_b:
        st.markdown("**Concept B**")
        q_b = st.text_input("Search concept B", placeholder='"Digital Lending", "Telemedicine"', key="uc3_qb")
        if st.button("Find B", key="uc3_find_b", disabled=not q_b):
            hits = api_get("/search-name", {"q": q_b, "limit": 10})
            if hits:
                st.session_state["uc3_hits_b"] = hits

        if st.session_state.get("uc3_hits_b"):
            opts_b = {f"{h['name']}  [{h['label']}]": h["node_id"] for h in st.session_state["uc3_hits_b"]}
            chosen_b = st.selectbox("Select B", list(opts_b.keys()), key="uc3_sel_b")
            st.session_state["uc3_node_b"] = opts_b[chosen_b]
            st.caption(f"`{st.session_state['uc3_node_b']}`")

    # ── Compare button ─────────────────────────────────────────────────
    node_a = st.session_state.get("uc3_node_a")
    node_b = st.session_state.get("uc3_node_b")
    can_compare = bool(node_a and node_b and node_a != node_b)

    if st.button("Compare", type="primary", disabled=not can_compare, key="uc3_compare"):
        with st.spinner("Computing similarity..."):
            result = api_post("/compare", {"node_id_a": node_a, "node_id_b": node_b})

        if result is None:
            st.stop()

        st.divider()

        # ── Similarity score ───────────────────────────────────────────
        sim_pct   = result["similarity_pct"]
        interp    = result["interpretation"]
        interp_color = {"High": "#1d9e75", "Medium": "#ba7517", "Low": "#d85a30"}[interp]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div style="text-align:center;padding:12px 0">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:40px;font-weight:500;color:{interp_color}">'
                f'{sim_pct}%</div>'
                f'<div style="font-size:12px;color:#888780">cosine similarity</div>'
                f'<div style="font-size:13px;font-weight:500;color:{interp_color};margin-top:4px">'
                f'{interp} similarity</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m2:
            dist = result["graph_distance"]
            dist_display = str(dist) + " hops" if dist >= 0 else "unreachable"
            st.markdown(
                f'<div style="text-align:center;padding:12px 0">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:40px;font-weight:500;color:#534ab7">'
                f'{dist_display}</div>'
                f'<div style="font-size:12px;color:#888780">graph distance</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m3:
            raw_score = result["cosine_similarity"]
            st.markdown(
                f'<div style="text-align:center;padding:12px 0">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:40px;font-weight:500;color:#444">'
                f'{raw_score}</div>'
                f'<div style="font-size:12px;color:#888780">raw cosine score</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        render_score_bar(result["cosine_similarity"])
        st.divider()

        # ── Side-by-side paths ─────────────────────────────────────────
        pc_a, pc_b = st.columns(2)
        with pc_a:
            st.markdown("**Concept A path**")
            render_path_crumb(result["node_a"]["path_str"])
        with pc_b:
            st.markdown("**Concept B path**")
            render_path_crumb(result["node_b"]["path_str"])

        # ── Graph path ─────────────────────────────────────────────────
        st.divider()
        graph_path = result.get("graph_path", [])
        if graph_path:
            st.markdown("**Shortest path through knowledge graph**")
            path_html = ""
            for i, name in enumerate(graph_path):
                path_html += f'<span style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#444">{name}</span>'
                if i < len(graph_path) - 1:
                    path_html += '<span style="color:#b4b2a9;font-size:14px;margin:0 6px">→</span>'
            st.markdown(path_html, unsafe_allow_html=True)
            st.caption(f"{len(graph_path) - 1} hops through the graph")
        else:
            st.caption("These two nodes are not reachable from each other in the graph.")

        # ── Interpretation guide ────────────────────────────────────────
        st.divider()
        st.markdown("**How to read this**")
        st.markdown(
            "| Score | Meaning |\n|---|---|\n"
            "| 75–100% | Concepts are semantically very close — may overlap or be near-duplicates |\n"
            "| 45–74%  | Related concepts in the same general area |\n"
            "| < 45%   | Distinct concepts — little semantic overlap |"
        )


# ══════════════════════════════════════════════════════════════════════
# UC5 — TAXONOMY COVERAGE AUDIT
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### Audit taxonomy coverage and detect semantic overlaps")
    st.caption(
        "See which dimensions have the most/fewest areas, and find concepts "
        "that are semantically similar but live in different parts of the taxonomy."
    )

    audit_tab_a, audit_tab_b = st.tabs(["📊 Coverage", "🔍 Overlap Detection"])

    # ── Sub-tab A: Coverage bar chart ──────────────────────────────────
    with audit_tab_a:
        if st.button("Load coverage data", type="primary", key="uc5_load"):
            with st.spinner("Fetching coverage stats..."):
                coverage_data = api_get("/coverage")
            if coverage_data:
                st.session_state["uc5_coverage"] = coverage_data

        if st.session_state.get("uc5_coverage"):
            cov = st.session_state["uc5_coverage"]

            # Group by domain for colour coding
            domain_colors = {
                "FinTech":      "#534ab7",
                "HealthTech":   "#1d9e75",
                "Insurtech":    "#ba7517",
                "RegTech":      "#d85a30",
                "TechTech":     "#888780",
                "RetailTech":   "#0f6e56",
                "InfoTech":     "#3c3489",
                "InfoServTech": "#633806",
            }

            # Render as horizontal bars using pure HTML/CSS (no charting library needed)
            max_count = max(c["area_count"] for c in cov) if cov else 1
            st.markdown(f"**{len(cov)} dimensions** across all domains")
            st.divider()

            # Domain filter
            all_domains = sorted(set(c["domain_name"] for c in cov))
            selected_domains = st.multiselect(
                "Filter by domain", all_domains, default=all_domains, key="uc5_domain_filter"
            )

            filtered = [c for c in cov if c["domain_name"] in selected_domains]

            for c in filtered:
                bar_pct = int((c["area_count"] / max_count) * 100)
                domain  = c["domain_name"]
                color   = domain_colors.get(domain, "#888780")
                st.markdown(
                    f'<div style="margin-bottom:10px">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                    f'<span style="font-size:13px;color:#333">{c["dimension_name"]}</span>'
                    f'<div>'
                    f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{color};'
                    f'background:{color}18;padding:1px 7px;border-radius:10px;margin-right:6px">{domain}</span>'
                    f'<span style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#888780">'
                    f'{c["area_count"]} areas</span>'
                    f'</div></div>'
                    f'<div style="background:#f1f0e8;border-radius:4px;height:8px">'
                    f'<div style="width:{bar_pct}%;background:{color};border-radius:4px;height:8px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            # Summary metrics
            st.divider()
            total_areas = sum(c["area_count"] for c in filtered)
            avg_areas   = total_areas // len(filtered) if filtered else 0
            thin_dims   = [c for c in filtered if c["area_count"] <= 3]

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total areas (filtered)", total_areas)
            mc2.metric("Avg areas / dimension", avg_areas)
            mc3.metric("Thin dimensions (≤3 areas)", len(thin_dims))

            if thin_dims:
                st.markdown("**Thin dimensions** (may need expansion):")
                pills = "".join(
                    f'<span class="node-pill pill-dim">{c["dimension_name"]} ({c["area_count"]})</span>'
                    for c in thin_dims
                )
                st.markdown(pills, unsafe_allow_html=True)

    # ── Sub-tab B: Overlap detection ───────────────────────────────────
    with audit_tab_b:
        st.markdown(
            "Finds area nodes that are **semantically close** (high cosine similarity) "
            "but live in **different dimensions** — potential duplicates or taxonomy gaps."
        )

        col_thresh, col_max = st.columns([2, 1])
        with col_thresh:
            threshold = st.slider(
                "Similarity threshold", min_value=0.70, max_value=0.98,
                value=0.82, step=0.01, key="uc5_threshold",
                help="Higher = stricter. Only pairs above this score are shown.",
            )
        with col_max:
            max_pairs = st.selectbox("Max pairs", [25, 50, 100], index=1, key="uc5_maxpairs")

        if st.button("Find overlaps", type="primary", key="uc5_overlaps"):
            with st.spinner(f"Scanning {threshold:.0%} similarity threshold across all area pairs..."):
                overlap_data = api_get("/overlaps", {"threshold": threshold, "max_pairs": max_pairs})
            if overlap_data is not None:
                st.session_state["uc5_overlaps"] = overlap_data

        if "uc5_overlaps" in st.session_state:
            pairs = st.session_state["uc5_overlaps"]

            if not pairs:
                st.info(f"No overlapping pairs found above {threshold:.0%} threshold. Try lowering it.")
            else:
                cross_dim = [p for p in pairs if not p["same_dimension"]]
                same_dim  = [p for p in pairs if p["same_dimension"]]

                st.markdown(
                    f"**{len(pairs)} pairs** found · "
                    f"**{len(cross_dim)} cross-dimension** (potential issues) · "
                    f"**{len(same_dim)} same-dimension** (expected)"
                )
                st.divider()

                show_filter = st.radio(
                    "Show", ["Cross-dimension only", "All pairs"], horizontal=True, key="uc5_filter"
                )
                display_pairs = cross_dim if show_filter == "Cross-dimension only" else pairs

                for p in display_pairs:
                    sim_pct   = p["similarity_pct"]
                    same      = p["same_dimension"]
                    badge_txt = "same dimension" if same else "⚠ cross-dimension"
                    badge_col = "#e1f5ee" if same else "#faeeda"
                    badge_txt_col = "#085041" if same else "#633806"

                    with st.container():
                        st.markdown(
                            f'<div style="border:0.5px solid #e5e3db;border-radius:10px;'
                            f'padding:12px 16px;margin-bottom:10px;background:#fafaf8">'
                            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
                            f'<span style="font-family:IBM Plex Mono,monospace;font-size:18px;'
                            f'font-weight:500;color:#1d9e75">{sim_pct}%</span>'
                            f'<div style="flex:1;background:#f1f0e8;border-radius:3px;height:4px">'
                            f'<div style="width:{sim_pct}%;background:#1d9e75;border-radius:3px;height:4px"></div></div>'
                            f'<span style="font-size:11px;padding:2px 8px;border-radius:10px;'
                            f'background:{badge_col};color:{badge_txt_col};'
                            f'font-family:IBM Plex Mono,monospace">{badge_txt}</span>'
                            f'</div>'
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
                            f'color:#534ab7;margin-bottom:4px">{p["node_a"]["path_str"]}</div>'
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
                            f'color:#0f6e56">{p["node_b"]["path_str"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                if cross_dim:
                    st.divider()
                    export_data = [
                        {
                            "path_a":     p["node_a"]["path_str"],
                            "path_b":     p["node_b"]["path_str"],
                            "similarity": p["cosine_similarity"],
                            "same_dim":   p["same_dimension"],
                        }
                        for p in display_pairs
                    ]
                    st.download_button(
                        "Export overlaps as JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name="taxonomy_overlaps.json",
                        mime="application/json",
                    )