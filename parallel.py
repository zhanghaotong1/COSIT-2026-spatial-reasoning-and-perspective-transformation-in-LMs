import os
import json
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


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


def test_model(rank, args, model, tokenizer):
    model.eval()

    test_dataset = SpatialData(args.test)
    test_dataloader = DataLoader(test_dataset, args.batch, shuffle=False)

    all_pred = []
    all_label = []

    with torch.no_grad():
        for texts, labels in tqdm(test_dataloader):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: inputs[k].to(rank) for k in inputs.keys()}
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


def train_model(rank, args, logger, has_model=False):
    train_dataset = SpatialData(args.train)
    train_dataloader = DataLoader(train_dataset, args.batch, shuffle=False, sampler=distributed.DistributedSampler(train_dataset))

    tokenizer, model = load_encoder_tokenizer(args)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    total_steps = len(train_dataloader) * args.epoch
    num_warmup_steps = int(0.1 * total_steps)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

    best_acc = 50

    if has_model:
        map_location = {'cuda:0': 'cuda:%d' % rank}
        checkpoint = torch.load(args.ckpt + '%s_%s_%s.pt' % (args.name, args.size, args.seed), map_location=map_location)
        ddp_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_acc = checkpoint['best_acc']

    for epoch_num in range(args.epoch):
        if rank == 0:
            logger.info('Start training epoch %s ...' % (epoch_num + 1))
        ddp_model.train()
        train_dataloader.sampler.set_epoch(epoch_num)

        for texts, labels in tqdm(train_dataloader):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: inputs[k].to(rank) for k in inputs.keys()}
            labels = torch.tensor(labels).to(rank)

            loss = ddp_model(**inputs, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if rank == 0:
            test_acc = test_model(rank, args, ddp_model.module, tokenizer)
            print('Epoch %s, test accuracy %.2f' % (epoch_num + 1, test_acc))
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({'model_state_dict': ddp_model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(), 'best_acc': best_acc}, args.ckpt + '%s_%s_%s.pt' % (args.name, args.size, args.seed))
                logger.info('Saving model to %s, epoch %s' % (args.ckpt + '%s_%s_%s.pt' % (args.name, args.size, args.seed), epoch_num + 1))
        dist.barrier()

    if rank == 0:
        print('Training finishes! Best accuracy on test set is %.2f' % best_acc)
    return


def main(rank, world_size, args, logger):
    path = args.ckpt + '%s_%s_%s.pt' % (args.name, args.size, args.seed)

    if args.task == 'train':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        if os.path.exists(path):
            logger.info('Start training %s-%s from checkpoint ...' % (args.name, args.size))
            train_model(rank, args, logger, has_model=True)
        else:
            logger.info('Start training %s-%s from pretrained model ...' % (args.name, args.size))
            train_model(rank, args, logger)

        dist.destroy_process_group()

    else:
        if os.path.exists(path):
            logger.info('Loading %s-%s from checkpoint ...' % (args.name, args.size))
            logger.info('Start testing %s-%s ...' % (args.name, args.size))
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            tokenizer, model = load_encoder_tokenizer(args)
            model = model.to(device)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            acc = test_model(device, args, model, tokenizer)
            print('Accuracy on test set is %.2f' % acc)
        else:
            print('No trained model. Please train first!')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='../data/train.txt', help='training set path')
    parser.add_argument('--test', type=str, default='../data/test.txt', help='test set path')
    parser.add_argument('--pred', type=str, default='../data/pred/', help='path to store predicted file')
    parser.add_argument('--ckpt', type=str, default='../model/', help='path to store checkpoint')
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test'], help='train or test')

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

    n_gpus = torch.cuda.device_count()
    logger.info('Using %d GPUs in total ...' % n_gpus)
    mp.spawn(main, args=(n_gpus, args, logger), nprocs=n_gpus)
