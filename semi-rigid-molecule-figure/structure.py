# %%
from automol import smiles

from learn_vrc_tst import geom

# %%
geo = smiles.geometry("CCC")
view = geom.view(geo)
view.write_html("visualization.html")
