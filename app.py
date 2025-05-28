import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from web_scraper import bfs, dfs, calculate_pagerank
import networkx as nx
import plotly.graph_objects as go
import numpy as np

some_colors = ['red', 'green', 'blue', 'yellow']

st.set_page_config(layout="wide")
st.title("🔍 Keyword Web Crawler with BFS & DFS")

url = st.text_input("Enter starting URL", "https://example.com")
keyword = st.text_input("Keyword to search for", "example")
max_depth = st.slider("Max crawl depth", 1, 3, 2)
visit_limit = st.slider("Max pages to visit", 5, 50, 20)

def visualize_graph(graph, pagerank_scores, title="Graph Visualization"):
    if not graph:
        st.warning("Graph is empty, nothing to visualize.")
        return

    G = nx.DiGraph()

    #Add nodes and edges for visualizations
    for node, edges in graph.items():
        size = max(pagerank_scores.get(node, 0) * 100 + 10, 10)
        G.add_node(node, size=size)
        for dest in edges:
            G.add_edge(node, dest)

    #Ensure all nodes have size attribute
    for node in G.nodes():
        if 'size' not in G.nodes[node]:
            G.nodes[node]['size'] = max(pagerank_scores.get(node, 0) * 100 + 10, 10)

    pos = nx.spring_layout(G, seed=42)

    #Prepare edge traces
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    #Prepare node traces
    node_x, node_y, node_size, node_text = [], [], [], []

    ranks = np.array([pagerank_scores.get(node, 0) for node in G.nodes()])
    if ranks.max() > ranks.min():
        norm_ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    else:
        norm_ranks = np.zeros_like(ranks)

    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        size = G.nodes[node]['size']
        node_size.append(size)
        node_text.append(f"{node}<br>PageRank: {pagerank_scores.get(node, 0):.6f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=norm_ranks,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(
                    text="PageRank",
                    side="right"
                )
            ),
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600,
                        width=900))

    st.plotly_chart(fig, use_container_width=True)

if st.button("Run BFS and DFS Crawlers"):
    st.info("Running both BFS and DFS crawlers...")

    with st.spinner("Running BFS..."):
        bfs_graph, bfs_hits, bfs_visited, bfs_time = bfs(url, keyword, max_depth, visit_limit)
        bfs_rank = calculate_pagerank(bfs_graph)

    with st.spinner("Running DFS..."):
        dfs_graph, dfs_hits, dfs_visited, dfs_time = dfs(url, keyword, max_depth, visit_limit)
        dfs_rank = calculate_pagerank(dfs_graph)

    st.header("Performance Comparison")
    perf_df = pd.DataFrame({
        'Method': ['BFS', 'DFS'],
        'Keyword Occurrences': [bfs_hits, dfs_hits],
        'Pages Visited': [bfs_visited, dfs_visited],
        'Time (s)': [bfs_time, dfs_time]
    })
    st.dataframe(perf_df)

    st.subheader("Visual Comparison")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].bar(['BFS', 'DFS'], [bfs_hits, dfs_hits], color=['blue', 'green'])
    ax[0].set_title("Keyword Hits")
    ax[1].bar(['BFS', 'DFS'], [bfs_visited, dfs_visited], color=['blue', 'green'])
    ax[1].set_title("Pages Visited")
    ax[2].bar(['BFS', 'DFS'], [bfs_time, dfs_time], color=['blue', 'green'])
    ax[2].set_title("Time Taken (s)")
    st.pyplot(fig)

    st.subheader("Top PageRanks (BFS)")
    bfs_df = pd.DataFrame(bfs_rank.items(), columns=["Page", "Rank"]).sort_values("Rank", ascending=False).head(5)
    st.dataframe(bfs_df)

    st.subheader("Top PageRanks (DFS)")
    dfs_df = pd.DataFrame(dfs_rank.items(), columns=["Page", "Rank"]).sort_values("Rank", ascending=False).head(5)
    st.dataframe(dfs_df)

    visualize_graph(bfs_graph, bfs_rank, "BFS Crawl Graph with PageRank")
    visualize_graph(dfs_graph, dfs_rank, "DFS Crawl Graph with PageRank")

# networkx plotly