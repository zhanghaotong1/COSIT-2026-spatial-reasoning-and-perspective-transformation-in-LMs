import json
from tqdm import tqdm
import random
import numpy as np
from scipy.optimize import minimize

nl2s = {'south': ('=', '<'),
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
    index3 = set()
    for f in formulae:
        index3.add((variables.index(f[1]) * 2, variables.index(f[0]) * 2, variables.index(f[2]) * 2 + 1, variables.index(f[0]) * 2 + 1,
                    variables.index(f[2]) * 2, variables.index(f[0]) * 2, variables.index(f[1]) * 2 + 1, variables.index(f[0]) * 2 + 1))
    index3 = list(index3)

    tds, labels = [], []
    for _ in range(8):
        vars1 = random.sample(variables, 2)
        vars2 = random.sample(variables, 2)
        if set(vars1) != set(vars2):
            fs = [(random.choice(list(nl2s.keys())), vars1[0], vars1[1]), (random.choice(list(nl2s.keys())), vars2[0], vars2[1])]
            index1, index2 = [], []

            for f in fs:
                r1, r2 = nl2s[f[0]]
                if r1 == '=':
                    index2.append((variables.index(f[1]) * 2, variables.index(f[2]) * 2))
                elif r1 == '<':
                    index1.append((variables.index(f[2]) * 2, variables.index(f[1]) * 2))
                else:
                    index1.append((variables.index(f[1]) * 2, variables.index(f[2]) * 2))

                if r2 == '=':
                    index2.append((variables.index(f[1]) * 2 + 1, variables.index(f[2]) * 2 + 1))
                elif r2 == '<':
                    index1.append((variables.index(f[2]) * 2 + 1, variables.index(f[1]) * 2 + 1))
                else:
                    index1.append((variables.index(f[1]) * 2 + 1, variables.index(f[2]) * 2 + 1))

            constraints = [{'type': 'ineq', 'fun': _lambda1(i)} for i in index1] + [{'type': 'eq', 'fun': _lambda2(i)} for i in index2] + [{'type': 'ineq', 'fun': _lambda3(i)} for i in index3]
            res = minimize(obj, x0=np.random.rand(len(variables) * 2), method='SLSQP', constraints=constraints)
            tds.append(fs)
            labels.append(res.success)

    return tds, labels


def logic2nl(td):
    return ' If %s is to the %s of %s, then %s is to the %s of %s.' % (td[0][1], td[0][0], td[0][2], td[1][1], td[1][0], td[1][2])


f = open('../2domr_train.txt')
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
            variables.add(t[12])
            variables.add(t[25])
            variables.add(t[28])
            if 'left' in t:
                formulae.append((t[12], t[25], t[28]))
            else:
                formulae.append((t[25], t[12], t[28]))

        variables = list(variables)
        assert len(variables) == ss['variables']
        assert len(formulae) == ss['formulae']

        tds, labels = satisfiability_checking(variables, formulae)
        for t, l in zip(tds, labels):
            if l:
                sat.append({'text': ss['text'] + logic2nl(t), 'label': 'satisfiable', 'variables': ss['variables'], 'id': ss['id']})
            else:
                unsat.append({'text': ss['text'] + logic2nl(t), 'label': 'unsatisfiable', 'variables': ss['variables'], 'id': ss['id']})

n = min(len(sat), len(unsat))
data = random.sample(sat, n) + random.sample(unsat, n)
random.shuffle(data)

f = open('../omr22d_train.txt', 'w')
for d in data:
    print(json.dumps(d), file=f)
f.close()
