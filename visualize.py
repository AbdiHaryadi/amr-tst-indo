from amrlib.graph_processing.amr_plot import AMRPlot

def render(graph_str: str, render_path: str):
    entry = loads_graph_only_amr_entry(graph_str)

    render_path_for_amrplot = render_path
    if render_path_for_amrplot.endswith(".png"):
        render_path_for_amrplot = render_path_for_amrplot[:-4]
    plot = AMRPlot(render_path_for_amrplot, format="png")

    plot.build_from_graph(entry, debug=False)
    actual_render_path = plot.render()
    return actual_render_path

def loads_graph_only_amr_entry(data):
    # Modified from amrlib.graph_processing.amr_loading.load_amr_entries
    lines = [l for l in data.splitlines() if not (l.startswith('#') and not \
                l.startswith('# ::'))]
    cleaned_data = '\n'.join(lines)
    entry = cleaned_data.strip()
    assert entry != "", "Cannot load empty graph."
    return entry
