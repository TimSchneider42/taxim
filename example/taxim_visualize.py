#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from taxim import Taxim, CALIB_GELSIGHT_MINI

# Create Taxim simulator
taxim = Taxim(calib_folder=CALIB_GELSIGHT_MINI, backend="auto")

# Generate some arbitrary heightmap
height_map = np.zeros((taxim.height, taxim.width))
height_map[200:300, 200:500] = -10.0

# Render an image using this height map
img = taxim.render(height_map, with_shadow=True, press_depth=1.0)

plt.imshow(img)
plt.show()
