from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure


def plot_graph(df, v):
    output_file(v["datadir"] + f"plot/{v['i']}_plot_pso_{v['mode']}_te{v['TE']}_se{v['SE']}.html")

    width, height = 350, 250
    fig1 = figure(width=width, plot_height=height, title="θ")
    fig2 = figure(width=width, plot_height=height, title="dθ")
    fig3 = figure(width=width, plot_height=height, title="ddθ")
    fig4 = figure(width=width, plot_height=height, title="trq")
    fig5 = figure(width=width, plot_height=height, title="w1")
    fig6 = figure(width=width, plot_height=height, title="w2")

    fig1.line(df["t"], df["θ"])
    fig2.line(df["t"], df["dθ"])
    fig3.line(df["t"], df["ddθ"])
    fig4.line(df["t"], df["trq"])
    fig5.line(df["t"], df["w1"])
    fig6.line(df["t"], df["w2"])

    fig = gridplot([[fig1, fig4], [fig2, fig5], [fig3, fig6],])

    if v["isShow"]:
        show(fig)
