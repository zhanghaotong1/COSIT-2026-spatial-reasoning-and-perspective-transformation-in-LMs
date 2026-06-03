import json
import random
import string
from collections import defaultdict
import networkx as nx
from networkx.exception import NetworkXError


def connectivity_checking(formulae):
    # Determine if the graph constructed based on the sample satisfies the connectivity constraint
    g = nx.Graph()
    for e in formulae:
        g.add_edge(e[1], e[2])
    return nx.is_connected(g)


def unsat_complexity(formulae):
    g = nx.DiGraph()
    edges_relation = defaultdict(list)
    for e in formulae:
        if e[0] == '=':
            edges_relation[(e[1], e[2])].append('=')
            g.add_edge(e[1], e[2])
            edges_relation[(e[2], e[1])].append('=')
            g.add_edge(e[2], e[1])
        elif e[0] == '>':
            edges_relation[(e[1], e[2])].append('>')
            g.add_edge(e[1], e[2])
        else:
            edges_relation[(e[2], e[1])].append('>')
            g.add_edge(e[2], e[1])

    cycle_len = []
    for c in nx.simple_cycles(g):
        c.append(c[0])
        # find the corresponding relation of each edge along the cycle
        relations_along_c = [edges_relation[(c[i], c[i+1])] for i in range(len(c) - 1)]
        if any(['>' in l for l in relations_along_c]):  # this cycle contains at least one '>'
            cycle_len.append(len(c) - 1)  # record the length of this cycle

    return min(cycle_len)


def satisfiability_checking(variable_num, formulae):
    # Check whether a sample is satisfiable.
    g = nx.MultiDiGraph()
    weighted_edges = []
    for e in formulae:
        if e[0] == '=':
            weighted_edges.append((e[1], e[2], 0))
            weighted_edges.append((e[2], e[1], 0))
        elif e[0] == '>':
            weighted_edges.append((e[1], e[2], -1))
        else:
            weighted_edges.append((e[2], e[1], -1))
    g.add_weighted_edges_from(weighted_edges)
    labels = []
    for i in range(variable_num):
        try:
            nx.find_negative_cycle(g, i)
            labels.append(False)
        except NetworkXError:
            labels.append(True)
    if all(labels):
        return True
    else:
        return False


def reverse(variable_num, formulae):
    for i, f in enumerate(formulae):
        if f[0] == '<':
            newformulae = formulae[:i] + [('>', f[1], f[2])] + formulae[i + 1:]
            label = satisfiability_checking(variable_num, newformulae)
            if label:
                return newformulae
        elif f[0] == '>':
            newformulae = formulae[:i] + [('<', f[1], f[2])] + formulae[i + 1:]
            label = satisfiability_checking(variable_num, newformulae)
            if label:
                return newformulae
        else:
            continue
    return None


def logic2nl(variables, formulae):
    # Transform formal formulas into natural language. It returns a string.
    nl = ''
    for f in formulae:
        if f[0] == '=':
            nl += variables[f[1]] + ' overlaps with ' + variables[f[2]] + '. '
        elif f[0] == '<':
            nl += variables[f[1]] + ' is on the left of ' + variables[f[2]] + '. '
        else:
            nl += variables[f[1]] + ' is on the right of ' + variables[f[2]] + '. '
    return nl.strip()


def data_generator(variables, variable_num, sample_size):
    vs = random.sample(variables, variable_num)
    selected_v = set()
    formulae = set()

    while len(formulae) < sample_size:
        p = random.choice(['=', '<', '>'])
        vps = random.sample(vs, 2)
        formulae.add((p, vps[0], vps[1]))
        selected_v.add(vps[0])
        selected_v.add(vps[1])
    formulae = [(f[0], vs.index(f[1]), vs.index(f[2])) for f in formulae]

    if len(selected_v) == variable_num and connectivity_checking(formulae) and not satisfiability_checking(variable_num, formulae):
        counter = reverse(variable_num, formulae)
        if counter:
            comp = unsat_complexity(formulae)
            unsat = {'text': logic2nl(vs, formulae), 'label': 'unsatisfiable', 'complexity': comp, 'variables': variable_num, 'formulae': sample_size}
            sat = {'text': logic2nl(vs, counter), 'label': 'satisfiable', 'complexity': comp, 'variables': variable_num, 'formulae': sample_size}
            return unsat, sat
        else:
            return None, None
    else:
        return None, None


variables = list(string.ascii_uppercase)
data = []
for i in range(3, 27):
    samples = []
    while len(samples) < 5000:
        unsat, sat = data_generator(variables, i, i)
        if unsat:
            samples.append(unsat)
            samples.append(sat)
    data.extend(samples)
    print(i)

f = open('../1d_total.txt', 'w')
for d in data:
    print(json.dumps(d), file=f)
f.close()
