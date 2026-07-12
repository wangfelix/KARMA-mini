"""Presentation helpers for the bounded Neo4j evidence subgraph."""


def _dot_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _shorten(value, max_chars):
    value = str(value or "")
    if len(value) <= max_chars:
        return value
    return value[:max_chars - 1].rstrip() + "…"


def subgraph_to_dot(subgraph):
    """Convert a ``{nodes, edges}`` payload to a safe Graphviz DOT string."""
    seed_names = set(subgraph.get("seed_names", []))
    lines = [
        "digraph Evidence {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.65"];',
        '  node [shape="box", style="rounded,filled", fontname="Arial", fontsize="10", color="#64748B", margin="0.10,0.06"];',
        '  edge [fontname="Arial", fontsize="9", color="#64748B", fontcolor="#334155", arrowsize="0.7"];',
    ]

    for node in subgraph.get("nodes", []):
        node_id = _dot_escape(node["id"])
        name = _shorten(node.get("label") or "Unnamed entity", 48)
        label_parts = [_dot_escape(name)]
        if node.get("paper_id"):
            label_parts.append(_dot_escape(_shorten(node["paper_id"], 40)))
        label = "\\n".join(label_parts)
        if node.get("label") in seed_names:
            fill = "#FEF3C7"
            border = "#D97706"
            penwidth = "2.0"
        elif node.get("paper_id"):
            fill = "#DBEAFE"
            border = "#2563EB"
            penwidth = "1.2"
        else:
            fill = "#F8FAFC"
            border = "#64748B"
            penwidth = "1.0"
        lines.append(
            f'  "{node_id}" [label="{label}", fillcolor="{fill}", '
            f'color="{border}", penwidth="{penwidth}"];'
        )

    for edge in subgraph.get("edges", []):
        source = _dot_escape(edge["source"])
        target = _dot_escape(edge["target"])
        label = _dot_escape(_shorten(edge.get("label") or edge.get("type"), 36))
        details = " | ".join(
            str(value) for value in (edge.get("paper_id"), edge.get("info_unit"))
            if value
        )
        tooltip = _dot_escape(details or edge.get("type") or "relationship")
        lines.append(
            f'  "{source}" -> "{target}" [label="{label}", tooltip="{tooltip}"];'
        )

    lines.append("}")
    return "\n".join(lines)
