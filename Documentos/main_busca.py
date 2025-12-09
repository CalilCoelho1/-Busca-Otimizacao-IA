import numpy as np
import matplotlib.pyplot as plt
from busca_discreta import GlobalRandomSearch, LocalRandomSearch


origem = [7,3]
quantidade_pontos = 50

pontos = np.vstack((
    origem,
    np.random.uniform(-10,15,size=(quantidade_pontos,2))
))

grs = GlobalRandomSearch(10000, pontos)
grs.search()
lrs = LocalRandomSearch(10000,pontos,2)
lrs.search()

plt.show()
bp=1