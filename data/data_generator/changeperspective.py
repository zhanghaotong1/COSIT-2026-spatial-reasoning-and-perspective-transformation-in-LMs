import json
import random
import string
import networkx as nx

nl2s = {'overlap': ('=', '='),
        'south': ('=', '<'),
        'north': ('=', '>'),
        'east': ('>', '='),
        'west': ('<', '='),
        'southeast': ('>', '<'),
        'southwest': ('<', '<'),
        'northeast': ('>', '>'),
        'northwest': ('<', '>')}


def connectivity(variables, formulae_survey, formulae_route):
    g = nx.Graph()
    for f in formulae_survey:
        g.add_edge(variables.index(f[1]), variables.index(f[2]))
    for f in formulae_route:
        g.add_edge(variables.index(f[0]), variables.index(f[1]))
        g.add_edge(variables.index(f[1]), variables.index(f[2]))
    return nx.is_connected(g)


def logic2nl(formulae_survey, formulae_omr):
    # Transform formal formulas into natural language. It returns a string.
    nl = []
    for f in formulae_survey:
        if f[0] == 'overlap':
            nl.append(f[1] + ' overlaps with ' + f[2] + '.')
        else:
            nl.append(f[1] + ' is to the ' + f[0] + ' of ' + f[2] + '.')

    for f in formulae_omr:
        if random.random() > 0.5:
            nl.append('Standing at %s and facing %s, %s is on the left.' % (f[0], f[1], f[2]))
        else:
            nl.append('Standing at %s and facing %s, %s is on the right.' % (f[1], f[0], f[2]))

    random.shuffle(nl)
    return ' '.join(nl)


def data_generator(variables, variable_num, survey_size, route_size):
    vs = random.sample(variables, variable_num)
    selected_v = set()
    formulae1 = set()
    formulae2 = set()

    while len(formulae1) < survey_size:
        p = random.choice(['overlap', 'south', 'north', 'east', 'west', 'southeast', 'southwest', 'northeast', 'northwest'])
        vps = random.sample(vs, 2)
        formulae1.add((p, vps[0], vps[1]))
        selected_v = selected_v.union(set(vps))
    formulae1 = list(formulae1)

    while len(formulae2) < route_size:
        vps = random.sample(vs, 3)
        formulae2.add((vps[0], vps[1], vps[2]))
        selected_v = selected_v.union(set(vps))
    formulae2 = list(formulae2)

    if len(selected_v) == variable_num and connectivity(vs, formulae1, formulae2):
        expr = 'Print[TimeConstrained[FindInstance['
        for f in formulae1:
            r1, r2 = nl2s[f[0]]
            if r1 == '=':
                expr += '%s1 == %s1 && ' % (f[1], f[2])
            else:
                expr += '%s1 %s %s1 && ' % (f[1], r1, f[2])
            if r2 == '=':
                expr += '%s2 == %s2 && ' % (f[1], f[2])
            else:
                expr += '%s2 %s %s2 && ' % (f[1], r2, f[2])
        for f in formulae2:
            expr += '(%s1 - %s1) * (%s2 - %s2) - (%s1 - %s1) * (%s2 - %s2) > 0 && ' % (f[1], f[0], f[2], f[0], f[2], f[0], f[1], f[0])
        var = ', '.join([v + '1, ' + v + '2' for v in vs])
        expr = expr.strip(' && ') + ', {' + var + '}, Reals], 0.5]]'
        return expr, logic2nl(formulae1, formulae2)
    else:
        return None, None


variables = list(string.ascii_uppercase)
f1 = open('/Users/user/Desktop/file.wls', 'w')
print('#!/usr/bin/env wolframscript', file=f1)
f2 = open('../cp.txt', 'w')
for i, m in zip(list(range(16)), [5100 for _ in range(16)]):
    n = 0
    while n < m:
        expr, nl = data_generator(variables, 15, i, 15 - i)
        if expr:
            print(expr, file=f1)
            print(json.dumps({'text': nl, 'survey': i, 'route': 15 - i}), file=f2)
            n += 1
    print(i)
f1.close()
f2.close()

# how to execute the wolframscript in the command line?
# chmod +x file.wls
# ./file.wls > output.txt
