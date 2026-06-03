import json

m, n = 0, 0
with open('omr22d_pred.txt') as f:
    for line in f:
        n += 1
        line = json.loads(line)
        pred = line['pred'].replace('*', ' ').strip().split()[-1].lower()
        if line['label'] == 'satisfiable' and pred == 'yes':
            m += 1
        elif line['label'] == 'unsatisfiable' and pred == 'no':
            m += 1
print(m / n)
