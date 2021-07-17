import inspect
from functools import partial

from graphviz import Digraph


class DataPipeline(object):
    ""
    def __init__(self, name, backend, run_pipeline):
        self.pipline = []
        self.dot = Digraph(comment=name)

    def add_node(self, func, name, preset_values={}):
        self.pipline.append(_Node(func=func, name=name, preset_values=preset_values))
        # add node to graph for visualization
        self.dot.node(name, label=name)
        
        if len(self.pipline) > 0:
            last_node = self.pipeline[-1]
            self.dot.edge(last_node.name, name)

    def render_graph(self, gv_filename):
        return self.dot.render(gv_filename, view=True)
        
    def __call__(self, *args, **kwargs):
        result = self.pipline[0](*args, **kwargs)
        for node in self.pipeline[1:]:
            result = node(results)
        return result


class _Node(object):
    def __init__(self, func, node_type, name=None, preset_values={}):
        self.sig = inspect.signature(func)
        # if a customized name is provided, else get string name from func 
        if name:
            self.name = name
        else:
            self.name = func.__name__  
        self.func = partial(func, **preset_values)
        self.node_type = node_type

    @property
    def signatures(self):
        signatures = parameter.name for parameter in self.sig
        return signatures

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def run_pipeline_dask(input_dict, pipeline):
    df = pipeline[0](input_dict['input_node'])
    for node in pipeline[1:-1]:
        df = df.map_partitions(node)
    
    if input_dict['output_node']['write']:
        if input_dict['output_node']['output_dir']:
       



if __name__ == '__main__':
    data_pipeline = DataPipeline(config={})
    
    data = data_pipeline(data, ch_log='error', fh_log='debug')
