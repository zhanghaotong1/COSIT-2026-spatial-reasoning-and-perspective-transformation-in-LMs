import os
import json
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='../data/train.txt', help='training set path')
parser.add_argument('--test', type=str, default='../data/test.txt', help='test set path')
parser.add_argument('--pred', type=str, default='../data/pred/', help='path to store predicted file')
parser.add_argument('--ckpt', type=str, default='../model/', help='path to store checkpoint')
parser.add_argument('--task', type=str, default='train', help='train, test or pretrain')

parser.add_argument('-n', '--name', type=str, required=True, help='language model')
parser.add_argument('-s', '--size', type=str, required=True, help='model size')
parser.add_argument('-b', '--batch', type=int, default=24, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=10, help='training epoch')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class SpatialData(Dataset):
    def __init__(self, datapath):
        f = open(datapath)
        datalist = f.readlines()
        f.close()
        self.data = [json.loads(d) for d in datalist]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]['text'], self.data[idx]['label']
        label = 0 if label == 'satisfiable' else 1
        return text.strip(), label


def load_encoder_tokenizer(args):
    # Load pretrained model and tokenizer
    if args.size not in ['base', 'large']:
        raise ValueError('Please choose a size from base or large!')
    if args.name == 't5':
        if args.size == 'base':
            tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
            model = AutoModelForSequenceClassification.from_pretrained('google-t5/t5-base')
        else:
            tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large')
            model = AutoModelForSequenceClassification.from_pretrained('google-t5/t5-large')
    elif args.name == 'deberta':
        if args.size == 'base':
            tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')
        else:
            tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large')
    elif args.name == 'roberta':
        if args.size == 'base':
            tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
            model = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-base')
        else:
            tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-large')
            model = AutoModelForSequenceClassification.from_pretrained('FacebookAI/roberta-large')
    elif args.name == 'bert':
        if args.size == 'base':
            tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
            model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased')
        else:
            tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-large-uncased')
            model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-large-uncased')
    elif args.name == 'gpt2':
        if args.size == 'base':
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained('openai-community/gpt2')
            model.config.pad_token_id = model.config.eos_token_id
        else:
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large')
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained('openai-community/gpt2-large')
            model.config.pad_token_id = model.config.eos_token_id
    elif args.name == 'xlnet':
        if args.size == 'base':
            tokenizer = AutoTokenizer.from_pretrained('xlnet/xlnet-base-cased')
            model = AutoModelForSequenceClassification.from_pretrained('xlnet/xlnet-base-cased')
        else:
            tokenizer = AutoTokenizer.from_pretrained('xlnet/xlnet-large-cased')
            model = AutoModelForSequenceClassification.from_pretrained('xlnet/xlnet-large-cased')
    elif args.name == 'bart':
        if args.size == 'base':
            tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
            model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-base')
        else:
            tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
            model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large')
    else:
        raise ValueError('Please choose a name from t5, deberta, roberta, bert, gpt2, xlnet or bart!')
    return tokenizer, model


def test_model(args, model, tokenizer, device):
    model.eval()

    test_dataset = SpatialData(args.test)
    test_dataloader = DataLoader(test_dataset, args.batch, shuffle=False)

    all_pred = []
    all_label = []

    with torch.no_grad():
        for texts, labels in tqdm(test_dataloader):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: inputs[k].to(device) for k in inputs.keys()}
            outputs = model(**inputs).logits

            all_pred.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())
            all_label.extend(labels)

    if args.task == 'test':
        data_file = open(args.test)
        data = data_file.readlines()
        data_file.close()
        assert len(data) == len(all_pred)

        pred_file = args.pred + f'{args.name}_{args.size}_{args.seed}.txt'
        f = open(pred_file, 'w')
        for d, p in zip(data, all_pred):
            d = json.loads(d.strip())
            d['pred'] = 'satisfiable' if p == 0 else 'unsatisfiable'
            print(json.dumps(d), file=f)
        f.close()

    return accuracy_score(all_label, all_pred) * 100


def train_model(args, device, logger, has_model=False):
    train_dataset = SpatialData(args.train)
    train_dataloader = DataLoader(train_dataset, args.batch, shuffle=True)

    tokenizer, model = load_encoder_tokenizer(args)
    model = model.to(device)

    total_steps = len(train_dataloader) * args.epoch
    num_warmup_steps = int(0.1 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

    best_acc = 50

    if has_model:
        checkpoint = torch.load(args.ckpt + '%s_%s_%s.pt' % (args.name, args.size, args.seed))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_acc = checkpoint['best_acc']

    for epoch_num in range(args.epoch):
        logger.info('Start training epoch %s ...' % (epoch_num + 1))
        model.train()
        avg_loss = 0.0

        for texts, labels in tqdm(train_dataloader):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: inputs[k].to(device) for k in inputs.keys()}
            labels = torch.tensor(labels).to(device)

            loss = model(**inputs, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item()

        test_acc = test_model(args, model, tokenizer, device)
        logger.info('Epoch %s, average loss %.2f, test accuracy %.2f' % (epoch_num + 1, avg_loss / len(train_dataloader), test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc}, args.ckpt + '%s_%s_%s.pt' % (args.name, args.size, args.seed))
            logger.info('Saving model to %s, epoch %s' % (args.ckpt + '%s_%s_%s.pt' % (args.name, args.size, args.seed), epoch_num + 1))

    print('Training finishes! Best accuracy on test set is %.2f' % best_acc)
    return


def train_classifier(args, device, logger, tolerance=5, max_epoch=50):
    train_dataset = SpatialData(args.train)
    train_dataloader = DataLoader(train_dataset, args.batch, shuffle=True)

    tokenizer, model = load_encoder_tokenizer(args)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    if args.name in ['t5', 'bart']:
        for param in model.classification_head.parameters():
            param.requires_grad = True
    elif args.name == 'deberta':
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.pooler.parameters():
            param.requires_grad = True
    elif args.name in ['roberta', 'bert']:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif args.name == 'gpt2':
        for param in model.score.parameters():
            param.requires_grad = True
    elif args.name == 'xlnet':
        for param in model.logits_proj.parameters():
            param.requires_grad = True
        for param in model.sequence_summary.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_acc = 50
    tol = 0

    for epoch_num in range(max_epoch):
        logger.info('Start training epoch %s ...' % (epoch_num + 1))
        model.train()
        avg_loss = 0.0

        for texts, labels in tqdm(train_dataloader):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: inputs[k].to(device) for k in inputs.keys()}
            labels = torch.tensor(labels).to(device)

            loss = model(**inputs, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        test_acc = test_model(args, model, tokenizer, device)
        logger.info('Epoch %s, average loss %.2f, test accuracy %.2f' % (epoch_num + 1, avg_loss / len(train_dataloader), test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            tol = 0
        else:
            tol += 1
        if tol >= tolerance:
            print('Accuracy on test set stop increasing, stop training ... highest accuracy is %.2f' % best_acc)
            return

    print('Training finishes! Best accuracy on test set is %.2f' % best_acc)
    return


device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
logger.info('Using device: %s' % device)

path = args.ckpt + '%s_%s_%s.pt' % (args.name, args.size, args.seed)
if args.task == 'train':
    if os.path.exists(path):
        logger.info('Start training %s-%s from checkpoint ...' % (args.name, args.size))
        train_model(args, device, logger, has_model=True)
    else:
        logger.info('Start training %s-%s from pretrained model ...' % (args.name, args.size))
        train_model(args, device, logger)
elif args.task == 'test':
    if os.path.exists(path):
        logger.info('Loading %s-%s from checkpoint ...' % (args.name, args.size))
        logger.info('Start testing %s-%s ...' % (args.name, args.size))
        tokenizer, model = load_encoder_tokenizer(args)
        model = model.to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        acc = test_model(args, model, tokenizer, device)
        print('Accuracy on test set is %.2f' % acc)
    else:
        print('No trained model. Please train first!')
else:
    logger.info('Loading %s-%s from pretrained model ...' % (args.name, args.size))
    train_classifier(args, device, logger)
