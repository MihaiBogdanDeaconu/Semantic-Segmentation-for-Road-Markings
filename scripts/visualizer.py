from visdom import Visdom
import json

class Visualizer(object):
    """A wrapper for the Visdom visualization tool."""
    def __init__(self, port='13579', env='main', id=None):
        """
        Initializes the Visualizer.

        Args:
            port: The port for the Visdom server.
            env: The Visdom environment to use.
            id: An optional identifier to prepend to plot names.
        """
        self.vis = Visdom(port=port, env=env)
        self.id = id
        self.env = env
    
    def vis_scalar(self, name, x, y, opts=None):
        """
        Plots a scalar value (like loss) over time.

        Args:
            name: The name of the plot window.
            x: The x-axis value (e.g., iteration).
            y: The y-axis value (e.g., loss).
            opts: Optional dictionary of Visdom plot options.
        """
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        if self.id is not None:
            name = "[%s]" % self.id + name
        default_opts = {'title': name}
        if opts is not None:
            default_opts.update(opts)

        self.vis.line(X=x, Y=y, win=name, opts=default_opts, update='append')
    
    def vis_image(self, name, img, env=None, opts=None):
        """
        Displays an image in a Visdom window.

        Args:
            name: The name of the image window.
            img: The image tensor to display.
            env: The Visdom environment to use.
            opts: Optional dictionary of Visdom plot options.
        """
        if env is None:
            env = self.env
        if self.id is not None:
            name = "[%s]" % self.id + name
        default_opts = {'title': name}
        if opts is not None:
            default_opts.update(opts)
        self.vis.image(img=img, win=name, opts=default_opts, env=env)
    
    def vis_table(self, name, tbl, opts=None):
        """
        Displays a dictionary as an HTML table.

        Args:
            name: The name of the table window.
            tbl: The dictionary to display.
            opts: Optional dictionary of Visdom plot options.
        """
        tbl_str = "<table width=\"100%\"> "
        tbl_str += "<tr><th>Term</th><th>Value</th></tr>"
        for k, v in tbl.items():
            tbl_str += f"<tr><td>{k}</td><td>{v}</td></tr>"
        tbl_str += "</table>"

        default_opts = {'title': name}
        if opts is not None:
            default_opts.update(opts)
        self.vis.text(tbl_str, win=name, opts=default_opts)

if __name__ == '__main__':
    import numpy as np
    vis = Visualizer(port=35588, env='main')
    
    print("Visualizing a sample table...")
    tbl = {"lr": 0.01, "momentum": 0.9, "batch_size": 16}
    vis.vis_table("Hyperparameters", tbl)

    print("Visualizing a sample scalar plot...")
    for i in range(10):
        vis.vis_scalar(name='loss', x=i, y=np.random.rand() + (10 - i) / 10)
