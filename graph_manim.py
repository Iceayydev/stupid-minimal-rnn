import os
import csv
from manim import *

class GraphRnnData(Scene):
    def construct(self):
        ax = Axes(
            x_range = (0,100,10),
            y_range = (0,1000,100),
            x_length = 12,
            y_length = 8,
            tips = False,
            axis_config = {
                "include_numbers": True,
            }
        )
        ax.center()
        #self.play(DrawBorderThenFill(ax))
        self.add(ax)

        coords = []

        for item in os.listdir("data/loss/"):
            x = [] 
            y = []

            with open(f"data/loss/{item}", "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    
                    x.append(int(row["epoch"])+1)
                    y.append(float(row["loss"]))


            line = ax.plot_line_graph(x,y,line_color=WHITE,add_vertex_dots=False)
            #self.play(Create(line),run_time=0.1)
            self.add(line)