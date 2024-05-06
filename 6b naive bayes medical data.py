from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pylab as plt
from networkx.drawing.nx_pydot import graphviz_layout
model = BayesianNetwork([('Guest', 'Host'), ('Price', 'Host')])
cpd_guest = TabularCPD('Guest', 3, [[0.33], [0.33], [0.33]])
cpd_price = TabularCPD('Price', 3, [[0.33], [0.33], [0.33]])
cpd_host = TabularCPD('Host', 3, [[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
[0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
[0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]],
evidence=['Guest', 'Price'],evidence_card=[3, 3])
model.add_cpds(cpd_guest, cpd_price, cpd_host)
model.check_model()
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
posterior_p = infer.query(['Host'], evidence={'Guest': 2, 'Price': 2})
print(posterior_p)
pos = graphviz_layout(model, prog='dot')
nx.draw(model,pos)
plt.savefig('model.jpg')
plt.close()