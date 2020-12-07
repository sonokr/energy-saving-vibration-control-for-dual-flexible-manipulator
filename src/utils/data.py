from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure

from utils.utils import create_dirs


def plot_graph(df, cfg):
    create_dirs(cfg["DATA"]["DIR"] + "plot/")
    output_file(
        cfg["DATA"]["DIR"]
        + f"plot/{0}_plot_pso_{cfg['COMM']['MODE']}_\
te{cfg['CALC']['TE_str']}_se{cfg['CALC']['SE_str']}.html"
    )

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

    if cfg["COMM"]["PLOT"]:
        show(fig)
