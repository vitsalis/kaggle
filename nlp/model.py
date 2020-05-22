from absl import app, flags, logging

import torch as th
import numpy as np

import nlp
import transformers


flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 1, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('percent', 5, '')

FLAGS = flags.FLAGS

class SentimentClassifier(th.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(model)

    def forward(self, xb):
        mask = (xb != 0).float()
        res, = self.model(xb, mask)
        return res

def get_loss(losses, nums):
    return np.sum(np.multiply(losses, nums)) / np.sum(nums)

def get_dl(ds, batch_size):
    return th.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

def load_data():
    tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)
    def _tokenize(x):
        x['input_ids'] = tokenizer.batch_encode_plus(
                            x['text'],
                            max_length=FLAGS.seq_length,
                            pad_to_max_length=True)['input_ids']
        return x

    def _prepare_ds(split):
        ds = nlp.load_dataset('imdb', split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else f"{FLAGS.percent}%"}]')
        ds = ds.map(_tokenize, batched=True)
        ds.set_format(type='torch', columns=['input_ids', 'label'])
        return ds

    return map(_prepare_ds, ('train', 'test'))

def get_optimizer(model, lr, mom):
    return th.optim.Adam(model.parameters(), lr=lr)

def get_model(model):
    return SentimentClassifier(model)

def loss_batch(model, loss_func, data, target, optimizer=None):
    loss = loss_func(model(data), target).sum()
    if optimizer != None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), len(data)

def train(epochs, model, optimizer, loss_func, train_loader, valid_loader):
    for epoch in range(epochs):
        model.train()

        train_losses, train_nums = zip(*[
            loss_batch(model, loss_func, x['input_ids'], x['label'], optimizer)
                for x in train_loader
        ])

        model.eval()

        with th.no_grad():
            valid_losses, valid_nums = zip(*[
                loss_batch(model, loss_func, data, target)
                    for data, target in valid_loader
            ])

        train_loss = get_loss(train_losses, train_nums)
        valid_loss = get_loss(valid_losses, valid_nums)

        print ('Train Epoch: {}\tTrain Loss: {:.6f}\tValid Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

def main(*args, **kwargs):
    # download and load data
    train_ds, valid_ds = load_data()
    train_dl = get_dl(train_ds, FLAGS.batch_size)
    valid_dl = get_dl(valid_ds, FLAGS.batch_size)

    # get model
    model = get_model(FLAGS.model)
    opt = get_optimizer(model, FLAGS.lr, FLAGS.momentum)

    # use cross entropy loss
    loss_func = th.nn.CrossEntropyLoss(reduction='none')

    # train
    train(FLAGS.epochs, model, opt, loss_func, train_dl, valid_dl)

if __name__ == "__main__":
    app.run(main)
