#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import json

from os import listdir
from os.path import isfile, join

from itertools import cycle

import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from abc import ABC, abstractmethod

class Plotter(ABC):
    def __init__(self, base_path, title):
        self.base_path = base_path
        self.title = title
        self.colors = [
            '#1F77B4', # muted blue
            '#FF7F0E',  # safety orange
            '#2CA02C',  # cooked asparagus green
            '#D62728',  # brick red
            '#9467BD',  # muted purple
            '#8C564B',  # chestnut brown
            '#E377C2',  # raspberry yogurt pink
            '#7F7F7F',  # middle gray
            '#BCBD22',  # curry yellow-green
            '#17BECF'  # blue-teal
        ]
        super().__init__()
        
    @abstractmethod
    def str_to_latex(self, original):
        pass
        
    @abstractmethod
    def row_to_nice_name(self, row):
        pass

    def make_lines(self, df, method_name, color, show_legend, visible, metric, show_std=False):
        m_name = metric[0]
        x_axis = metric[1]
        y_axis = metric[2]
        dff = df[df["nice_name"] == method_name]
        if x_axis is not None:
            dff = dff.sort_values(by = [x_axis])
            x = np.unique(dff[x_axis])
            y_mean = []
            y_std = []
            for xi, score_df in dff.groupby([x_axis])["scores"]:
                score_dict = score_df.values[0]
                for m in y_axis:
                    score_dict = score_dict[m]
                y_mean.append(np.mean(score_dict))
                y_std.append(np.std(score_dict))
        else:
            y_mean = []
            y_std = []
            for score_dict in dff["scores"]:
                print(score_dict)
                for m in y_axis:
                    score_dict = score_dict[m]
                y_mean.append(np.mean(score_dict))
                y_std.append(np.std(score_dict))

        l_name = self.str_to_latex(method_name)
        if not show_std:
            y_std = []

        if x_axis is None:
            return go.Bar(x = [l_name], y=y_mean, name=l_name, marker=dict(color=color)) #
        else:
            return go.Scatter(x=x, y=y_mean, line=dict(color=color), name=l_name, showlegend = show_legend,
                              visible = visible, 
                              error_y = dict(type="data", array=y_std, visible=not bool(np.any(np.isnan(y_std))))
                            )

    def read_data(self, path):
        df = pd.read_json(path, lines=True)
        df = df.fillna("None")
        df["nice_name"] = df.apply(self.row_to_nice_name, axis=1)
        
        return df

    @st.cache(allow_output_mutation=True)
    def read_files(self, base_path):
        fnames = [f for f in listdir(base_path) if isfile(join(base_path, f)) and f.endswith("jsonl")]
        dfs = {}
        for fname in fnames:
            print("Reading " + join(base_path, fname))
            dfs[fname] = self.read_data(join(base_path, fname))

        return dfs

    def plot_selected(self, df, selected_configs, metrics, show_std=True):
        if len(metrics) == 1:
            cols = rows = 1
        else:
            cols = 2
            rows = int(len(metrics) / cols) 
            if len(metrics) % cols != 0:
                rows += 1

        # TODO CHECK IF METRICS EXIST BEFOREHAND
        titles = [m for m, _, _ in metrics]
        fig = make_subplots(rows=rows,cols=cols,subplot_titles=titles, vertical_spacing=0.1)

        if len(selected_configs) == 0:
            for r in range(1,rows+1):
                for c in range(1,cols+1):
                    fig.append_trace(go.Scatter(x=[],y=[]),row = r,col = c)
        else:
            for method,color in zip(selected_configs, cycle(self.colors)):
                visible = True
                r = 1
                c = 1

                for m in metrics:
                    first = (c == 1 and r == 1)
                    fig.append_trace(self.make_lines(df, method, color, first, visible, m, show_std),row=r,col=c)
                    c += 1
                    if c > 2:
                        c = 1
                        r += 1

        r = 1
        c = 1
        for m in metrics:
            fig.update_xaxes(title_text=m[1], row=r, col=c)
            #fig.update_yaxes(title_text='_'.join(mi for mi in m[2]), row=r, col=c)
            #fig.update_yaxes(title_text="", row=r, col=c)
            c += 1
            if c > 2:
                c = 1
                r += 1

        return fig

    def run(self, plot_metrics):
        st.title(self.title)
        dfs = self.read_files(self.base_path)
        selected_df = st.selectbox('Select results file to display',list(dfs.keys()))
        df = dfs[selected_df]

        st.subheader("Loaded " + selected_df)
        st.sidebar.subheader("Select configurations to plot")
        show_raw = st.checkbox('Show raw entries', value=False)
        if show_raw:
            st.subheader('Raw results')
            st.write(df)

        all_configs = np.unique(df["nice_name"]).tolist()
        selected_configs = []
        for cfg_name,color in zip(all_configs, cycle(self.colors)):
            agree = st.sidebar.checkbox(cfg_name, value=False)
            if agree:
                selected_configs.append(cfg_name)

        show_std = st.sidebar.checkbox('Show error bars', value=False)
        fig = go.Figure(self.plot_selected(df, selected_configs, plot_metrics, show_std))
        show_legend = st.sidebar.checkbox('Show legend entries', value=False)
        fig.update_layout(height=1200, showlegend=show_legend, legend=dict(x=0,y=-0.1), legend_orientation="h")
        st.plotly_chart(fig, height=1200)

        #save_to_pdf = st.button("Store plots as PDF")
        # if save_to_pdf:
        #     pgf_template = """
        # \\documentclass[tikz,border=5pt]{standalone}

        # \\usepackage{xcolor}
        # \\usepackage{pgfplots}
        # \\pgfplotsset{compat=newest}
        # {colors}

        # \\begin{document}
        # \\begin{tikzpicture}
        #     \\begin{axis}[
        #         xlabel={xlabel},
        #         ylabel={ylabel},
        #         legend pos=north west
        #     ]

        # {plots}

        #     \\end{axis}
        # \\end{tikzpicture}
        # \\end{document}
        #     """
        #     pgf_path = self.base_path + "/" + selected_df.split(".csv")[0] 
        #     for m in plot_metrics:
        #         pgf_path_metric = pgf_path + "_" + m + ".tex"
        #         x_axis = "K"
        #         y_axis = m
        #         plot_str = ""

        #         for method,color in zip(selected_configs, cycle(colors)): 
        #             dff = df[df["nice_name"] == method]
        #             x = np.unique(dff[x_axis])
        #             if y_axis + "_std" in list(dff):
        #                 y_mean, y_std = dff.groupby([x_axis])[y_axis].mean(), dff.groupby([x_axis])[y_axis + "_std"].mean()
        #             else:
        #                 y_mean, y_std = dff.groupby([x_axis])[y_axis].mean(), dff.groupby([x_axis])[y_axis].std()

        #             coord = "\n\t\t\t".join(["({}, {})".format(xi,yi) for xi, yi in zip(x,y_mean)])
        #             plot_str += "\t\t\\addplot[mark=none,color={color}] coordinates { {coord} }; \n\t\t\\addlegendentry{{legend_name}} \n".replace("{coord}", coord).replace("{legend_name}", method).replace("{color}", color.replace("#",""))

        #         color_str = "\n".join(["\\definecolor{{value}}{HTML}{{value}}".replace("{value}", color.replace("#","")) for color in colors])
        #         pgf_str = pgf_template.replace("{colors}", color_str).replace("{xlabel}", x_axis).replace("{ylabel}", y_axis).replace("{plots}", plot_str)
        #         with open(pgf_path_metric, "w") as f:
        #             comments_str = "%" + comments.replace("\n", "\n%")
        #             f.write(comments_str + "\n" + pgf_str)

        #         st.text("PGF file written to {}".format(pgf_path_metric))