import json
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, help='ckpt path')
parser.add_argument('--data', type=str, nargs='+', help='data path (can input many data files)')
parser.add_argument('--key', type=str, choices=['c', 'f'], help='complexity or formulae')
parser.add_argument('--batch', type=int, default=24, help='batch size')
args = parser.parse_args()


class SpatialData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]['text'], self.data[idx]['label']
        label = 0 if label == 'satisfiable' else 1
        return text.strip(), label


def test_model(datalist, model, tokenizer, batchsize, device):
    model.eval()

    dataset = SpatialData(datalist)
    dataloader = DataLoader(dataset, batchsize, shuffle=False)

    all_pred = []
    all_label = []

    with torch.no_grad():
        for texts, labels in tqdm(dataloader):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: inputs[k].to(device) for k in inputs.keys()}
            outputs = model(**inputs).logits

            all_pred.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
            all_label.extend(labels)

    return accuracy_score(all_label, all_pred) * 100


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large')
model = model.to(device)
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state_dict'])

whole = []
for f in args.data:
    file = open(f)
    whole.extend(file.readlines())
    file.close()

if args.key == 'c':
    for i in range(3, 13):  # range (3,16) for 2D task
        data = []
        for ss in whole:
            ss = json.loads(ss.strip())
            if ss['complexity'] == i:
                data.append(ss)
        acc = test_model(data, model, tokenizer, args.batch, device)
        print('complexity %d, accuracy %.2f' % (i, acc))

else:
    for i in range(4, 27):
        data = []
        for ss in whole:
            ss = json.loads(ss.strip())
            if ss['formulae'] == i:
                data.append(ss)
        acc = test_model(data, model, tokenizer, args.batch, device)
        print('formulae %d, accuracy %.2f' % (i, acc))
