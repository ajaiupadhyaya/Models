"""
D3.js Visualizations for Financial Data
Advanced interactive visualizations using D3.js via JavaScript bridge
"""

import json
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import js2py
    HAS_JS2PY = True
except ImportError:
    HAS_JS2PY = False
    logger.warning("js2py not available. D3.js visualizations will use fallback to Plotly.")


class D3Visualizations:
    """
    Create D3.js-powered visualizations for financial data.
    Generates HTML/JavaScript code that can be embedded in dashboards.
    """
    
    def __init__(self):
        """Initialize D3 visualizations."""
        self.d3_version = "7.8.5"
        self.has_js_bridge = HAS_JS2PY
    
    def _get_d3_template(self) -> str:
        """Get D3.js library template."""
        return f"""
        <script src="https://d3js.org/d3.v{self.d3_version.split('.')[0]}.min.js"></script>
        <style>
            .d3-chart {{
                font-family: 'Inter', sans-serif;
                background: #0a0e27;
                color: #e5e7eb;
            }}
            .axis {{
                font-size: 12px;
                fill: #9ca3af;
            }}
            .axis path,
            .axis line {{
                fill: none;
                stroke: #374151;
                shape-rendering: crispEdges;
            }}
            .grid line {{
                stroke: #374151;
                stroke-opacity: 0.3;
            }}
            .line {{
                fill: none;
                stroke-width: 2px;
            }}
            .area {{
                fill-opacity: 0.3;
            }}
            .tooltip {{
                position: absolute;
                background: rgba(21, 27, 61, 0.95);
                border: 1px solid #1f2937;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                pointer-events: none;
                color: #e5e7eb;
            }}
        </style>
        """
    
    def candlestick_chart_d3(self,
                            data: pd.DataFrame,
                            width: int = 1200,
                            height: int = 600,
                            title: str = "Price Action") -> str:
        """
        Create D3.js candlestick chart.
        
        Args:
            data: DataFrame with OHLCV data
            width: Chart width
            height: Chart height
            title: Chart title
        
        Returns:
            HTML string with embedded D3.js visualization
        """
        # Prepare data
        chart_data = []
        for idx, row in data.iterrows():
            chart_data.append({
                'date': str(idx),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row.get('Volume', 0))
            })
        
        data_json = json.dumps(chart_data)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            {self._get_d3_template()}
        </head>
        <body>
            <div id="candlestick-chart" class="d3-chart"></div>
            <script>
                const data = {data_json};
                
                const margin = {{top: 40, right: 50, bottom: 60, left: 80}};
                const chartWidth = {width} - margin.left - margin.right;
                const chartHeight = {height} - margin.top - margin.bottom;
                
                const svg = d3.select("#candlestick-chart")
                    .append("svg")
                    .attr("width", {width})
                    .attr("height", {height});
                
                const g = svg.append("g")
                    .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
                
                // Scales
                const xScale = d3.scaleBand()
                    .domain(data.map(d => d.date))
                    .range([0, chartWidth])
                    .padding(0.1);
                
                const yScale = d3.scaleLinear()
                    .domain([d3.min(data, d => d.low) * 0.98, d3.max(data, d => d.high) * 1.02])
                    .range([chartHeight, 0]);
                
                // Color scale
                const colorScale = d3.scaleOrdinal()
                    .domain([true, false])
                    .range(["#10b981", "#ef4444"]);
                
                // Candlesticks
                const candles = g.selectAll(".candle")
                    .data(data)
                    .enter()
                    .append("g")
                    .attr("class", "candle")
                    .attr("transform", d => `translate(${{xScale(d.date)}},0)`);
                
                // Wicks
                candles.append("line")
                    .attr("x1", xScale.bandwidth() / 2)
                    .attr("x2", xScale.bandwidth() / 2)
                    .attr("y1", d => yScale(d.high))
                    .attr("y2", d => yScale(d.low))
                    .attr("stroke", d => colorScale(d.close >= d.open))
                    .attr("stroke-width", 1);
                
                // Bodies
                candles.append("rect")
                    .attr("x", 0)
                    .attr("y", d => yScale(Math.max(d.open, d.close)))
                    .attr("width", xScale.bandwidth())
                    .attr("height", d => Math.abs(yScale(d.close) - yScale(d.open)))
                    .attr("fill", d => colorScale(d.close >= d.open))
                    .attr("stroke", d => colorScale(d.close >= d.open));
                
                // Axes
                const xAxis = d3.axisBottom(xScale)
                    .tickFormat(d => d3.timeFormat("%m/%d")(new Date(d)))
                    .ticks(10);
                
                const yAxis = d3.axisLeft(yScale)
                    .tickFormat(d => `${{d.toFixed(2)}}`);
                
                g.append("g")
                    .attr("class", "axis")
                    .attr("transform", `translate(0,${{chartHeight}})`)
                    .call(xAxis);
                
                g.append("g")
                    .attr("class", "axis")
                    .call(yAxis);
                
                // Title
                svg.append("text")
                    .attr("x", {width} / 2)
                    .attr("y", 25)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#e5e7eb")
                    .attr("font-size", "18px")
                    .attr("font-weight", "600")
                    .text("{title}");
                
                // Tooltip
                const tooltip = d3.select("body").append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);
                
                candles.on("mouseover", function(event, d) {{
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    tooltip.html(`
                        <strong>${{d.date}}</strong><br/>
                        Open: ${{d.open.toFixed(2)}}<br/>
                        High: ${{d.high.toFixed(2)}}<br/>
                        Low: ${{d.low.toFixed(2)}}<br/>
                        Close: ${{d.close.toFixed(2)}}
                    `)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                }})
                .on("mouseout", function(d) {{
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
                }});
            </script>
        </body>
        </html>
        """
        
        return html
    
    def force_directed_network(self,
                               nodes: List[Dict],
                               links: List[Dict],
                               width: int = 1000,
                               height: int = 800,
                               title: str = "Network Analysis") -> str:
        """
        Create D3.js force-directed network graph.
        Useful for correlation networks, relationship mapping.
        
        Args:
            nodes: List of node dictionaries with 'id' and optional 'group', 'value'
            links: List of link dictionaries with 'source', 'target', 'value'
            width: Chart width
            height: Chart height
            title: Chart title
        
        Returns:
            HTML string with D3.js visualization
        """
        nodes_json = json.dumps(nodes)
        links_json = json.dumps(links)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            {self._get_d3_template()}
        </head>
        <body>
            <div id="network-chart" class="d3-chart"></div>
            <script>
                const nodes = {nodes_json};
                const links = {links_json};
                
                const svg = d3.select("#network-chart")
                    .append("svg")
                    .attr("width", {width})
                    .attr("height", {height});
                
                // Color scale
                const color = d3.scaleOrdinal(d3.schemeCategory10);
                
                // Simulation
                const simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id).distance(d => 100 / (d.value || 1)))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter({width} / 2, {height} / 2));
                
                // Links
                const link = svg.append("g")
                    .selectAll("line")
                    .data(links)
                    .enter().append("line")
                    .attr("stroke", "#374151")
                    .attr("stroke-opacity", 0.6)
                    .attr("stroke-width", d => Math.sqrt(d.value || 1));
                
                // Nodes
                const node = svg.append("g")
                    .selectAll("circle")
                    .data(nodes)
                    .enter().append("circle")
                    .attr("r", d => (d.value || 5) + 5)
                    .attr("fill", d => color(d.group || 0))
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                
                // Labels
                const label = svg.append("g")
                    .selectAll("text")
                    .data(nodes)
                    .enter().append("text")
                    .text(d => d.id)
                    .attr("font-size", "12px")
                    .attr("fill", "#e5e7eb")
                    .attr("dx", 12)
                    .attr("dy", 4);
                
                simulation.on("tick", () => {{
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    node
                        .attr("cx", d => d.x)
                        .attr("cy", d => d.y);
                    
                    label
                        .attr("x", d => d.x)
                        .attr("y", d => d.y);
                }});
                
                function dragstarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
                
                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}
                
                function dragended(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
                
                // Title
                svg.append("text")
                    .attr("x", {width} / 2)
                    .attr("y", 25)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#e5e7eb")
                    .attr("font-size", "18px")
                    .attr("font-weight", "600")
                    .text("{title}");
            </script>
        </body>
        </html>
        """
        
        return html
    
    def sankey_diagram(self,
                      nodes: List[str],
                      links: List[Dict],
                      width: int = 1200,
                      height: int = 800,
                      title: str = "Flow Diagram") -> str:
        """
        Create D3.js Sankey diagram for flow visualization.
        Useful for capital flows, DCF breakdowns, etc.
        
        Args:
            nodes: List of node names
            links: List of link dictionaries with 'source', 'target', 'value'
            width: Chart width
            height: Chart height
            title: Chart title
        
        Returns:
            HTML string with D3.js Sankey visualization
        """
        nodes_json = json.dumps([{"name": n} for n in nodes])
        links_json = json.dumps(links)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            {self._get_d3_template()}
            <script src="https://cdn.jsdelivr.net/npm/d3-sankey@0.12.2/dist/d3-sankey.min.js"></script>
        </head>
        <body>
            <div id="sankey-chart" class="d3-chart"></div>
            <script>
                const nodes = {nodes_json};
                const links = {links_json};
                
                const svg = d3.select("#sankey-chart")
                    .append("svg")
                    .attr("width", {width})
                    .attr("height", {height});
                
                const sankey = d3.sankey()
                    .nodeWidth(15)
                    .nodePadding(10)
                    .extent([[1, 1], [{width} - 1, {height} - 6]]);
                
                const {{graph}} = sankey({{
                    nodes: nodes.map(d => Object.assign({{}}, d)),
                    links: links.map(d => Object.assign({{}}, d))
                }});
                
                // Links
                const link = svg.append("g")
                    .selectAll(".link")
                    .data(graph.links)
                    .enter().append("path")
                    .attr("class", "link")
                    .attr("d", d3.sankeyLinkHorizontal())
                    .attr("stroke-width", d => Math.max(1, d.width))
                    .style("fill", "none")
                    .style("stroke", "#374151")
                    .style("stroke-opacity", 0.5);
                
                // Nodes
                const node = svg.append("g")
                    .selectAll(".node")
                    .data(graph.nodes)
                    .enter().append("g")
                    .attr("class", "node");
                
                node.append("rect")
                    .attr("x", d => d.x0)
                    .attr("y", d => d.y0)
                    .attr("height", d => d.y1 - d.y0)
                    .attr("width", d => d.x1 - d.x0)
                    .style("fill", "#1e3a8a")
                    .style("stroke", "#3b82f6");
                
                node.append("text")
                    .attr("x", d => d.x0 - 6)
                    .attr("y", d => (d.y1 + d.y0) / 2)
                    .attr("dy", "0.35em")
                    .attr("text-anchor", "end")
                    .text(d => d.name)
                    .style("fill", "#e5e7eb")
                    .style("font-size", "12px");
                
                // Title
                svg.append("text")
                    .attr("x", {width} / 2)
                    .attr("y", 25)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#e5e7eb")
                    .attr("font-size", "18px")
                    .attr("font-weight", "600")
                    .text("{title}");
            </script>
        </body>
        </html>
        """
        
        return html
    
    def treemap_d3(self,
                   data: Dict[str, float],
                   width: int = 1000,
                   height: int = 600,
                   title: str = "Treemap") -> str:
        """
        Create D3.js treemap visualization.
        
        Args:
            data: Dictionary of {name: value}
            width: Chart width
            height: Chart height
            title: Chart title
        
        Returns:
            HTML string with D3.js treemap
        """
        data_hierarchy = {
            "name": "root",
            "children": [{"name": k, "value": v} for k, v in data.items()]
        }
        data_json = json.dumps(data_hierarchy)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            {self._get_d3_template()}
        </head>
        <body>
            <div id="treemap-chart" class="d3-chart"></div>
            <script>
                const data = {data_json};
                
                const root = d3.hierarchy(data)
                    .sum(d => d.value)
                    .sort((a, b) => b.value - a.value);
                
                d3.treemap()
                    .size([{width}, {height}])
                    .padding(2)
                    (root);
                
                const svg = d3.select("#treemap-chart")
                    .append("svg")
                    .attr("width", {width})
                    .attr("height", {height});
                
                const color = d3.scaleOrdinal()
                    .domain(root.children.map(d => d.data.name))
                    .range(d3.schemeCategory10);
                
                const cell = svg.selectAll("g")
                    .data(root.leaves())
                    .enter().append("g")
                    .attr("transform", d => `translate(${{d.x0}},${{d.y0}})`);
                
                cell.append("rect")
                    .attr("width", d => d.x1 - d.x0)
                    .attr("height", d => d.y1 - d.y0)
                    .attr("fill", d => color(d.data.name))
                    .attr("stroke", "#1f2937");
                
                cell.append("text")
                    .attr("x", d => (d.x1 - d.x0) / 2)
                    .attr("y", d => (d.y1 - d.y0) / 2)
                    .attr("dy", "0.35em")
                    .attr("text-anchor", "middle")
                    .text(d => d.data.name)
                    .style("fill", "#fff")
                    .style("font-size", "12px");
                
                // Title
                svg.append("text")
                    .attr("x", {width} / 2)
                    .attr("y", 25)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#e5e7eb")
                    .attr("font-size", "18px")
                    .attr("font-weight", "600")
                    .text("{title}");
            </script>
        </body>
        </html>
        """
        
        return html
    
    def save_html(self, html_content: str, filepath: Path):
        """
        Save HTML visualization to file.
        
        Args:
            html_content: HTML string
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved D3.js visualization to {filepath}")


# Convenience function for integration with Dash/Plotly
def create_d3_iframe(html_content: str, width: int = 1200, height: int = 600) -> str:
    """
    Create iframe HTML for embedding D3.js visualizations in Dash.
    
    Args:
        html_content: HTML content from D3Visualizations methods
        width: Iframe width
        height: Iframe height
    
    Returns:
        HTML iframe string
    """
    import base64
    html_encoded = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
    
    return f'<iframe src="data:text/html;base64,{html_encoded}" width="{width}" height="{height}" frameborder="0"></iframe>'
