import json
import networkx as nx
from pyvis.network import Network
import os

def visualize_kg(json_path="data/output/final_kg.json", output_html="data/output/graph.html"):
    print(f"Loading Knowledge Graph from {json_path}...")

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run the pipeline first!")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create an empty directed graph
    G = nx.DiGraph()

    color_map = {
        "DRUG": "#FF9999",     # Red-ish
        "DISEASE": "#99CCFF",  # Blue-ish
        "PROTEIN": "#99FF99",  # Green-ish
        "GENE": "#FFCC99",     # Yellow-ish
        "CHEMICAL": "#FFB366", # Orange-ish
        "OTHER": "#E0E0E0"     # Gray
    }

    # 1. Add Nodes
    for entity in data.get('entities', []):
        ent_id = entity.get('entity_id') or entity.get('name')
        if not ent_id:
            continue # Skip invalid nodes

        ent_name = entity.get('name', str(ent_id))
        ent_type = entity.get('entity_type', 'OTHER')

        node_color = color_map.get(ent_type, color_map["OTHER"])

        G.add_node(
            ent_id,
            label=ent_name,
            title=f"Type: {ent_type}", # Hover tooltip
            group=ent_type,
            color=node_color
        )

    # 2. Add Edges (Relationships)
    for triple in data.get('triples', []):
        G.add_edge(
            triple['head'],
            triple['tail'],
            title=f"Confidence: {triple['confidence']:.2f}\nSource: {triple['source']}", # Hover tooltip
            label=triple['relation'],
            arrows="to"
        )


    print("Generating interactive HTML visualization...")
    net = Network(notebook=False, directed=True, width="100%", height="800px")
    net.force_atlas_2based()

    net.from_nx(G)

    # Save HTML file
    net.save_graph(output_html)
    print(f"Visualization saved successfully! Open '{output_html}' in your web browser.")

if __name__ == "__main__":
    visualize_kg()
