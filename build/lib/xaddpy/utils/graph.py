"""Provides the visualization tool using graphviz"""
import pygraphviz as pgv


class Graph(pgv.AGraph):
    def __init__(
            self, 
            thing=None, 
            filename=None, 
            data=None, 
            string=None, 
            handle=None, 
            name="", 
            strict=True, 
            directed=False, 
            **attr
    ):
        super().__init__(
            thing, filename, data, string, handle, name, strict, directed, **attr
        )
        self._node_to_label = {}
        self._node_to_color = {}
        self._node_to_style = {}
        self._node_to_shape = {}

    def add_node_label(self, node: str, label: str):
        self._node_to_label[node] = label
    
    def add_node_color(self, node: str, color: str):
        self._node_to_color[node] = color
        
    def add_node_shape(self, node: str, shape: str):
        self._node_to_shape[node] = shape
        
    def add_node_style(self, node: str, style: str):
        self._node_to_style[node] = style
        
    def configure(self):
        self.configure_graph_attributes(
            fontname="Helvetica",
            fontsize="16",
            ratio="auto",
            size="7.5,10",
            # rankdir="LR",
            ranksep="2.00"
        )
        self.configure_edge_attributes(
            fontsize="16"
        )
        self.configure_nodes()
        
    def configure_nodes(self):
        self.configure_node_attributes(
            fontsize="16",
        )
        for node in self.nodes():
            attrs = {}
            color = self._node_to_color.get(node)
            style = self._node_to_style.get(node)
            shape = self._node_to_shape.get(node)
            label = self._node_to_label.get(node)
            if label is not None:
                attrs['label'] = label
            if (color is not None) or (shape is not None) or (style is not None):
                if color is not None and color != '':
                    attrs['fillcolor'] = color
                    attrs['color'] = 'black'
                if shape is not None and shape != '':
                    attrs['shape'] = shape
                if style is not None and style != '' and color is not None:
                    attrs['style'] = style
            node.attr.update(attrs)
    
    def configure_edge_attributes(
            self,
            **attrs,
    ):
        self.edge_attr.update(attrs)
    
    def configure_node_attributes(
            self,
            **attrs,
    ):
        self.node_attr.update(attrs)
    
    def configure_graph_attributes(
            self,
            **attrs,
    ):
        self.graph_attr.update(attrs)    