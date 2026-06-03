import json
import random
import string
import networkx as nx
import numpy as np
from scipy.optimize import minimize


def connectivity(variables, formulae):
    # formulae is a list of tuples in the form (p1, p2, p3)
    formulae = [(variables.index(f[0]), variables.index(f[1]), variables.index(f[2])) for f in formulae]
    g = nx.Graph()
    for f in formulae:
        g.add_edge(f[0], f[1])
        g.add_edge(f[1], f[2])
    return nx.is_connected(g)


def satisfiability_scipy(variables, formulae):
    def obj(r):
        return 0

    def _lambda(i):
        return lambda r: (r[i[0]] - r[i[1]]) * (r[i[2]] - r[i[3]]) - (r[i[4]] - r[i[5]]) * (r[i[6]] - r[i[7]]) - 1e-6

    index = [(variables.index(f[1]) * 2, variables.index(f[0]) * 2, variables.index(f[2]) * 2 + 1, variables.index(f[0]) * 2 + 1,
              variables.index(f[2]) * 2, variables.index(f[0]) * 2, variables.index(f[1]) * 2 + 1, variables.index(f[0]) * 2 + 1) for f in formulae]
    res = minimize(obj, x0=np.random.rand(len(variables) * 2), method='SLSQP', constraints=[{'type': 'ineq', 'fun': _lambda(i)} for i in index])
    return res.success


def reverse(variables, formulae):
    for i, f in enumerate(formulae):
        newformulae = formulae[:i] + [(f[1], f[0], f[2])] + formulae[i + 1:]
        if satisfiability_scipy(variables, newformulae):
            return newformulae
    return None


def logic2nl(formulae):
    nl = ''
    for f in formulae:
        if random.random() > 0.5:
            nl += 'Standing at %s and facing %s, %s is on the left. ' % (f[0], f[1], f[2])
        else:
            nl += 'Standing at %s and facing %s, %s is on the right. ' % (f[1], f[0], f[2])
    return nl.strip()


def data_generator(variables, variable_num, sample_size):
    vs = random.sample(variables, variable_num)
    selected_v = set()
    formulae = set()

    while len(formulae) < sample_size:
        vps = random.sample(vs, 3)
        formulae.add((vps[0], vps[1], vps[2]))
        selected_v.add(vps[0])
        selected_v.add(vps[1])
        selected_v.add(vps[2])
    formulae = list(formulae)

    if len(selected_v) == variable_num and connectivity(vs, formulae) and not satisfiability_scipy(vs, formulae):
        counter = reverse(vs, formulae)
        if counter:
            unsat = {'text': logic2nl(formulae), 'label': 'unsatisfiable', 'variables': variable_num, 'formulae': sample_size}
            sat = {'text': logic2nl(counter), 'label': 'satisfiable', 'variables': variable_num, 'formulae': sample_size}
            return unsat, sat
        else:
            return None, None
    return None, None


variables = list(string.ascii_uppercase)
train, test = [], []
n = 0

for i in range(3, 27):
    samples = []
    while len(samples) < 3500:
        unsat, sat = data_generator(variables, i, i)
        if unsat:
            unsat['id'] = n
            sat['id'] = n
            samples.append(unsat)
            samples.append(sat)
            n += 1
    train.extend(samples[:3084])
    test.extend(samples[3084:])
    print(i)

random.shuffle(train)
random.shuffle(test)

f = open('../omr_train.txt', 'w')
for d in train:
    print(json.dumps(d), file=f)
f.close()
f = open('../omr_test.txt', 'w')
for d in test:
    print(json.dumps(d), file=f)
f.close()
