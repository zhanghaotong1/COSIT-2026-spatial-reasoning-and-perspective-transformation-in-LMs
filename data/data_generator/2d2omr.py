import json
from tqdm import tqdm
import random
import numpy as np
from scipy.optimize import minimize

nl2s = {'overlap': ('=', '='),
        'south': ('=', '<'),
        'north': ('=', '>'),
        'east': ('>', '='),
        'west': ('<', '='),
        'southeast': ('>', '<'),
        'southwest': ('<', '<'),
        'northeast': ('>', '>'),
        'northwest': ('<', '>')}


def satisfiability_checking(variables, formulae):
    def obj(r):
        return 0

    def _lambda1(i):
        return lambda r: r[i[0]] - r[i[1]] - 1e-6

    def _lambda2(i):
        return lambda r: r[i[0]] - r[i[1]]

    def _lambda3(i):
        return lambda r: (r[i[0]] - r[i[1]]) * (r[i[2]] - r[i[3]]) - (r[i[4]] - r[i[5]]) * (r[i[6]] - r[i[7]]) - 1e-6

    # sometimes SLSQP might incorrectly raise "Singular matrix C in LSQ subproblem" error when two constraints are exactly the same
    index1, index2 = set(), set()
    for f in formulae:
        r1, r2 = nl2s[f[0]]
        if r1 == '=':
            index2.add(tuple(sorted((variables.index(f[1]) * 2, variables.index(f[2]) * 2))))
        elif r1 == '<':
            index1.add((variables.index(f[2]) * 2, variables.index(f[1]) * 2))
        else:
            index1.add((variables.index(f[1]) * 2, variables.index(f[2]) * 2))

        if r2 == '=':
            index2.add(tuple(sorted((variables.index(f[1]) * 2 + 1, variables.index(f[2]) * 2 + 1))))
        elif r2 == '<':
            index1.add((variables.index(f[2]) * 2 + 1, variables.index(f[1]) * 2 + 1))
        else:
            index1.add((variables.index(f[1]) * 2 + 1, variables.index(f[2]) * 2 + 1))
    index1 = list(index1)
    index2 = list(index2)

    omrs, labels = [], []
    for _ in range(6):
        selected = random.sample(variables, 3)
        if ('overlap', selected[0], selected[1]) in formulae or ('overlap', selected[1], selected[0]) in formulae:
            continue
        index3 = [(variables.index(selected[1]) * 2, variables.index(selected[0]) * 2, variables.index(selected[2]) * 2 + 1, variables.index(selected[0]) * 2 + 1,
                   variables.index(selected[2]) * 2, variables.index(selected[0]) * 2, variables.index(selected[1]) * 2 + 1, variables.index(selected[0]) * 2 + 1)]
        constraints = [{'type': 'ineq', 'fun': _lambda1(i)} for i in index1] + [{'type': 'eq', 'fun': _lambda2(i)} for i in index2] + [{'type': 'ineq', 'fun': _lambda3(i)} for i in index3]
        res = minimize(obj, x0=np.random.rand(len(variables) * 2), method='SLSQP', constraints=constraints)
        omrs.append(selected)
        labels.append(res.success)

    return omrs, labels


def logic2nl(omr):
    return ' Standing at %s and facing %s, %s is on the left.' % (omr[0], omr[1], omr[2])


f = open('../2d_train.txt')
s = f.readlines()
f.close()

sat, unsat = [], []
for ss in tqdm(s, total=74000):
    ss = json.loads(ss.strip())
    if ss['label'] == 'satisfiable':
        text = ss['text'].strip('.').split('. ')
        formulae = []
        variables = set()
        for t in text:
            variables.add(t[0])
            variables.add(t[-1])
            if 'overlap' in t:
                formulae.append(('overlap', t[0], t[-1]))
            else:
                formulae.append((t.split()[4], t[0], t[-1]))

        variables = list(variables)
        assert len(variables) == ss['variables']
        assert len(formulae) == ss['formulae']

        omrs, labels = satisfiability_checking(variables, formulae)
        for o, l in zip(omrs, labels):
            if l:
                sat.append({'text': ss['text'] + logic2nl(o), 'label': 'satisfiable', 'variables': ss['variables'], 'id': ss['id']})
            else:
                unsat.append({'text': ss['text'] + logic2nl(o), 'label': 'unsatisfiable', 'variables': ss['variables'], 'id': ss['id']})

n = min(len(sat), len(unsat))
data = random.sample(sat, n) + random.sample(unsat, n)
random.shuffle(data)

f = open('../2d2omr_train.txt', 'w')
for d in data:
    print(json.dumps(d), file=f)
f.close()
