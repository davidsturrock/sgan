from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


fig, ax = plt.subplots()
for i, f in enumerate(sorted(Path(Path.cwd()).rglob(f'1674651*.txt'))):
    text = np.loadtxt(fname=f)
    x = [i[0] for i in text]
    y = [i[1] for i in text]

    ax.plot(x, y, linestyle='', marker='.', label=f'{f.name}')

plt.axis('square')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
plt.grid(which='both', axis='both', linestyle='-', linewidth=0.5)
ax.legend()
plt.show()
plt.close()