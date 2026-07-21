"""Interactive presentation helpers for the bounded Neo4j evidence graph."""

from html import escape


def _dot_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _shorten(value, max_chars):
    value = str(value or "")
    if len(value) <= max_chars:
        return value
    return value[:max_chars - 1].rstrip() + "…"


def subgraph_to_html(subgraph, height="620px"):
    """Render the evidence graph as a self-contained PyVis/vis-network page.

    Resources are embedded into the HTML so the graph remains interactive
    without loading scripts from a CDN. Importing PyVis lazily also lets the
    caller fall back to Graphviz if the optional renderer is unavailable.
    """
    from pyvis.network import Network

    seed_names = set(subgraph.get("seed_names", []))
    network = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#0F172A",
        cdn_resources="in_line",
    )

    for node in subgraph.get("nodes", []):
        name = str(node.get("label") or "Unnamed entity")
        paper_id = node.get("paper_id")
        if name in seed_names:
            color = {"background": "#FBBF24", "border": "#B45309"}
            size = 28
            group = "answer evidence"
        elif paper_id:
            color = {"background": "#60A5FA", "border": "#1D4ED8"}
            size = 24
            group = "paper-scoped structure"
        else:
            color = {"background": "#CBD5E1", "border": "#475569"}
            size = 21
            group = "shared content entity"

        tooltip = [f"<b>{escape(name)}</b>", f"Type: {escape(group)}"]
        if paper_id:
            tooltip.append(f"Paper: {escape(str(paper_id))}")
        network.add_node(
            node["id"],
            label=_shorten(name, 42),
            title="<br>".join(tooltip),
            color=color,
            size=size,
            shape="dot",
            borderWidth=2,
        )

    for edge in subgraph.get("edges", []):
        predicate = str(edge.get("label") or edge.get("type") or "related to")
        details = [f"<b>{escape(predicate)}</b>"]
        if edge.get("paper_id"):
            details.append(f"Paper: {escape(str(edge['paper_id']))}")
        if edge.get("info_unit"):
            details.append(f"Information unit: {escape(str(edge['info_unit']))}")
        network.add_edge(
            edge["source"],
            edge["target"],
            label=_shorten(predicate, 32),
            title="<br>".join(details),
            arrows="to",
            color="#64748B",
        )

    network.set_options("""
    {
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "hover": true,
        "keyboard": true,
        "navigationButtons": true,
        "tooltipDelay": 120,
        "zoomView": true
      },
      "nodes": {
        "font": {"face": "Arial", "size": 14, "color": "#0F172A"},
        "shadow": {"enabled": true, "color": "rgba(15, 23, 42, 0.16)", "size": 8, "x": 1, "y": 2}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.75}},
        "font": {
          "align": "middle",
          "background": "rgba(255,255,255,0.88)",
          "color": "#334155",
          "face": "Arial",
          "size": 11,
          "strokeWidth": 0
        },
        "smooth": {"enabled": true, "type": "dynamic"},
        "width": 1.4
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "avoidOverlap": 0.7,
          "centralGravity": 0.012,
          "damping": 0.45,
          "gravitationalConstant": -58,
          "springConstant": 0.055,
          "springLength": 155
        },
        "maxVelocity": 32,
        "minVelocity": 0.5,
        "stabilization": {"enabled": true, "iterations": 180, "updateInterval": 20}
      }
    }
    """)
    page = network.generate_html(notebook=False)
    fullscreen_controls = """
    <style>
      html, body { margin: 0; padding: 0; overflow: hidden; }
      #kg-fullscreen-button {
        align-items: center;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.14);
        color: #0F172A;
        cursor: pointer;
        display: flex;
        font: 600 13px Arial, sans-serif;
        gap: 7px;
        padding: 8px 11px;
        position: fixed;
        right: 12px;
        top: 12px;
        z-index: 1000;
      }
      #kg-fullscreen-button:hover { background: #F8FAFC; border-color: #94A3B8; }
      #kg-fullscreen-button:focus-visible { outline: 3px solid #93C5FD; outline-offset: 2px; }
      :fullscreen #mynetwork { height: 100vh !important; }
      html.expanded-window #mynetwork { height: 100vh !important; }
    </style>
    <button id="kg-fullscreen-button" type="button" title="Expand graph"
            aria-label="Expand graph">
      <span aria-hidden="true">⛶</span><span id="kg-fullscreen-label">Expand</span>
    </button>
    <script>
      (function () {
        const button = document.getElementById("kg-fullscreen-button");
        const label = document.getElementById("kg-fullscreen-label");

        function openExpandedWindow() {
          const popup = window.open("", "_blank");
          if (!popup) {
            console.error("Could not open the expanded graph window");
            return;
          }
          popup.document.open();
          popup.document.write("<!doctype html>" + document.documentElement.outerHTML);
          popup.document.close();
          popup.document.title = "GraphRAG evidence graph";
          popup.document.documentElement.classList.add("expanded-window");
          popup.setTimeout(function () {
            if (typeof popup.network !== "undefined") {
              popup.network.redraw();
              popup.network.fit({animation: {duration: 350, easingFunction: "easeInOutQuad"}});
            }
          }, 180);
        }

        button.addEventListener("click", async function () {
          if (!document.fullscreenElement &&
              typeof document.documentElement.requestFullscreen !== "function") {
            openExpandedWindow();
            return;
          }
          try {
            if (!document.fullscreenElement) {
              await document.documentElement.requestFullscreen();
            } else {
              await document.exitFullscreen();
            }
          } catch (error) {
            console.error("Could not toggle graph fullscreen mode", error);
            openExpandedWindow();
          }
        });

        document.addEventListener("fullscreenchange", function () {
          const active = Boolean(document.fullscreenElement);
          const buttonText = active ? "Exit full screen" : "Expand";
          label.textContent = buttonText;
          button.title = active ? "Exit full screen" : "Expand graph";
          button.setAttribute("aria-label", active ? "Exit full screen" : "Expand graph");
          setTimeout(function () {
            if (typeof network !== "undefined") {
              network.redraw();
              network.fit({animation: {duration: 350, easingFunction: "easeInOutQuad"}});
            }
          }, 120);
        });
      })();
    </script>
    """
    return page.replace("</body>", fullscreen_controls + "\n</body>")


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
