import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from collections import defaultdict, Counter
import json


class TaxonomyVisualizer:
    def __init__(self):
        self.tree_data = defaultdict(dict)
        self.flat_terms = []

    def parse_indexing_terms(self, indexing_terms_list):
        """Parse hierarchical indexing terms and build a 5-level classification tree"""
        for term in indexing_terms_list:
            if isinstance(term, str) and '->' in term:
                levels = [level.strip() for level in term.split('->')]

                # Ensure there are 5 levels, pad with empty strings if insufficient
                while len(levels) < 5:
                    levels.append('')

                # Take only the first 5 levels
                levels = levels[:5]

                level1, level2, level3, level4, level5 = levels

                # Build tree structure
                if level1:
                    if level1 not in self.tree_data:
                        self.tree_data[level1] = {'_count': 0}
                    self.tree_data[level1]['_count'] += 1

                    if level2:
                        if level2 not in self.tree_data[level1]:
                            self.tree_data[level1][level2] = {'_count': 0}
                        self.tree_data[level1][level2]['_count'] += 1

                        if level3:
                            if level3 not in self.tree_data[level1][level2]:
                                self.tree_data[level1][level2][level3] = {'_count': 0}
                            self.tree_data[level1][level2][level3]['_count'] += 1

                            if level4:
                                if level4 not in self.tree_data[level1][level2][level3]:
                                    self.tree_data[level1][level2][level3][level4] = {'_count': 0}
                                self.tree_data[level1][level2][level3][level4]['_count'] += 1

                                if level5:
                                    if level5 not in self.tree_data[level1][level2][level3][level4]:
                                        self.tree_data[level1][level2][level3][level4][level5] = 0
                                    self.tree_data[level1][level2][level3][level4][level5] += 1

                # Store flattened term information
                self.flat_terms.append({
                    'full_term': term,
                    'level1': level1,
                    'level2': level2,
                    'level3': level3,
                    'level4': level4,
                    'level5': level5
                })

    def create_sunburst_chart(self):
        """Create sunburst chart visualization"""
        labels = []
        parents = []
        values = []
        colors = []

        # Define color schemes
        color_schemes = {
            1: px.colors.qualitative.Set1,
            2: px.colors.qualitative.Set2,
            3: px.colors.qualitative.Set3,
            4: px.colors.qualitative.Pastel1,
            5: px.colors.qualitative.Pastel2
        }

        def add_nodes(data, parent_label, level):
            for key, value in data.items():
                if key == '_count':
                    continue

                current_label = f"{parent_label}/{key}" if parent_label else key

                if isinstance(value, dict):
                    if '_count' in value:
                        count = value['_count']
                    else:
                        count = sum(v if isinstance(v, int) else v.get('_count', 0)
                                  for k, v in value.items() if k != '_count')

                    labels.append(key)
                    parents.append(parent_label)
                    values.append(count)

                    # Assign colors
                    color_idx = len(labels) % len(color_schemes.get(level, px.colors.qualitative.Set1))
                    colors.append(color_schemes.get(level, px.colors.qualitative.Set1)[color_idx])

                    # Recursively add child nodes
                    add_nodes(value, key, level + 1)
                else:
                    # Leaf node
                    labels.append(key)
                    parents.append(parent_label)
                    values.append(value)

                    color_idx = len(labels) % len(color_schemes.get(level, px.colors.qualitative.Set1))
                    colors.append(color_schemes.get(level, px.colors.qualitative.Set1)[color_idx])

        # Add root node
        labels.append("Taxonomy Root Node")
        parents.append("")
        values.append(sum(data.get('_count', 0) for data in self.tree_data.values()))
        colors.append('#FFFFFF')

        add_nodes(self.tree_data, "Taxonomy Root Node", 1)

        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>',
            maxdepth=6
        ))

        fig.update_layout(
            title="Taxonomy Tree Structure - Sunburst Chart",
            font_size=12,
            width=800,
            height=800
        )

        return fig

    def create_treemap_chart(self):
        """Create treemap chart visualization"""
        df_rows = []

        def extract_hierarchy(data, path=[]):
            for key, value in data.items():
                if key == '_count':
                    continue

                current_path = path + [key]

                if isinstance(value, dict):
                    if '_count' in value:
                        count = value['_count']
                    else:
                        count = sum(v if isinstance(v, int) else v.get('_count', 0)
                                  for k, v in value.items() if k != '_count')

                    # Add current level
                    row = {
                        'ids': '/'.join(current_path),
                        'labels': key,
                        'parents': '/'.join(path) if path else '',
                        'values': count,
                        'level': len(current_path)
                    }
                    df_rows.append(row)

                    # Recursively process sub-levels
                    extract_hierarchy(value, current_path)
                else:
                    # Leaf node
                    row = {
                        'ids': '/'.join(current_path),
                        'labels': key,
                        'parents': '/'.join(path) if path else '',
                        'values': value,
                        'level': len(current_path)
                    }
                    df_rows.append(row)

        extract_hierarchy(self.tree_data)

        fig = go.Figure(go.Treemap(
            ids=[row['ids'] for row in df_rows],
            labels=[row['labels'] for row in df_rows],
            parents=[row['parents'] for row in df_rows],
            values=[row['values'] for row in df_rows],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Path: %{id}<extra></extra>'
        ))

        fig.update_layout(
            title="Taxonomy Tree Structure - Treemap Chart",
            font_size=12,
            width=800,
            height=600
        )

        return fig

    def create_network_graph(self):
        """Create network graph visualization"""
        G = nx.DiGraph()

        # Add nodes and edges
        for term_info in self.flat_terms:
            levels = [term_info[f'level{i}'] for i in range(1, 6) if term_info[f'level{i}']]

            # Add nodes
            for i, level in enumerate(levels):
                node_id = f"L{i+1}:{level}"
                if not G.has_node(node_id):
                    G.add_node(node_id, level=i+1, label=level)

            # Add edges
            for i in range(len(levels) - 1):
                parent = f"L{i+1}:{levels[i]}"
                child = f"L{i+2}:{levels[i+1]}"
                if G.has_edge(parent, child):
                    G[parent][child]['weight'] += 1
                else:
                    G.add_edge(parent, child, weight=1)

        # Use hierarchical layout
        pos = {}
        levels = defaultdict(list)
        for node, data in G.nodes(data=True):
            levels[data['level']].append(node)

        y_positions = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}

        for level, nodes in levels.items():
            y = y_positions[level]
            x_step = 8.0 / (len(nodes) + 1) if len(nodes) > 1 else 4.0
            for i, node in enumerate(nodes):
                pos[node] = (x_step * (i + 1), y)

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]]['weight'])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []

        color_map = {1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b'}

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node labels and colors
            level = G.nodes[node]['level']
            label = G.nodes[node]['label']
            node_text.append(f"L{level}: {label}")
            node_colors.append(color_map.get(level, '#1f77b4'))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                size=15,
                colorbar=dict(
                    thickness=15,
                    xanchor="left",
                    titleside="right"
                ),
                line=dict(width=2)
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Taxonomy Tree Structure - Network Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hierarchical classification network, different colors represent different levels",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#000000", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=1000,
                           height=600
                       ))

        return fig

    def generate_statistics(self):
        """Generate statistics information"""
        stats = {
            'total_terms': len(self.flat_terms),
            'level_distribution': defaultdict(int),
            'top_level1_categories': Counter(),
            'top_level2_categories': Counter(),
            'depth_distribution': Counter()
        }

        for term_info in self.flat_terms:
            # Count distribution of each level
            for i in range(1, 6):
                if term_info[f'level{i}']:
                    stats['level_distribution'][f'level{i}'] += 1

            # Count top-level categories
            if term_info['level1']:
                stats['top_level1_categories'][term_info['level1']] += 1
            if term_info['level2']:
                stats['top_level2_categories'][term_info['level2']] += 1

            # Count depth distribution
            depth = sum(1 for i in range(1, 6) if term_info[f'level{i}'])
            stats['depth_distribution'][depth] += 1

        return stats

    def create_statistics_charts(self):
        """Create statistical charts"""
        stats = self.generate_statistics()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Term Count Distribution by Level',
                'Depth Distribution',
                'Top 10 Level-1 Categories',
                'Top 10 Level-2 Categories'
            ),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Level distribution bar chart
        levels = list(stats['level_distribution'].keys())
        counts = list(stats['level_distribution'].values())
        fig.add_trace(
            go.Bar(x=levels, y=counts, name="Level Distribution"),
            row=1, col=1
        )

        # Depth distribution pie chart
        depths = list(stats['depth_distribution'].keys())
        depth_counts = list(stats['depth_distribution'].values())
        fig.add_trace(
            go.Pie(labels=[f'{d} Levels' for d in depths], values=depth_counts, name="Depth Distribution"),
            row=1, col=2
        )

        # Top 10 level-1 categories
        top_level1 = stats['top_level1_categories'].most_common(10)
        if top_level1:
            categories, counts = zip(*top_level1)
            fig.add_trace(
                go.Bar(x=list(categories), y=list(counts), name="Top-level Categories"),
                row=2, col=1
            )

        # Top 10 level-2 categories
        top_level2 = stats['top_level2_categories'].most_common(10)
        if top_level2:
            categories, counts = zip(*top_level2)
            fig.add_trace(
                go.Bar(x=list(categories), y=list(counts), name="Second-level Categories"),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Taxonomy Statistical Analysis"
        )

        return fig


def visualize_taxonomy_from_dataframe(df):
    """Create taxonomy visualization from DataFrame"""
    visualizer = TaxonomyVisualizer()

    # Parse indexing_terms column
    all_terms = []
    for terms in df['indexing_terms']:
        if isinstance(terms, str):
            all_terms.append(terms)
        elif isinstance(terms, list):
            all_terms.extend(terms)

    visualizer.parse_indexing_terms(all_terms)

    # Generate all visualization charts
    sunburst_fig = visualizer.create_sunburst_chart()
    treemap_fig = visualizer.create_treemap_chart()
    network_fig = visualizer.create_network_graph()
    stats_fig = visualizer.create_statistics_charts()

    return {
        'sunburst': sunburst_fig,
        'treemap': treemap_fig,
        'network': network_fig,
        'statistics': stats_fig,
        'visualizer': visualizer
    }


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Load data
    df = pd.read_pickle('../resources/kb_indexing.pkl')

    # Create visualization
    charts = visualize_taxonomy_from_dataframe(df)

    # Display charts
    charts['sunburst'].show()
    charts['treemap'].show()
    charts['network'].show()
    charts['statistics'].show()

    # Print statistics
    stats = charts['visualizer'].generate_statistics()
    print(f"Total terms: {stats['total_terms']}")
    print(f"Level distribution: {dict(stats['level_distribution'])}")
    print(f"Depth distribution: {dict(stats['depth_distribution'])}")
