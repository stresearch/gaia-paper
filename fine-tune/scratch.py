import holoviews as hv
hv.extension('bokeh')
out = hv.Curve((range(10), range(10))).opts(width=400, height=400)
hv.save(out,"plot_finetune.png")
