import sh
import nlp
import transformers
import torch as th
import pytorch_lightning as pl

from absl import logging, app, flags

class SentimentClassifier(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.train_ds = None
        self.valid_ds = None

        self.batch_size = params['batch_size']
        self.debug = params['debug']
        self.percent = params['percent']
        self.model_name = params['model']
        self.seq_length = params['seq_length']
        self.lr = params['lr']

        self.model = transformers.BertForSequenceClassification.from_pretrained(self.model_name)
        self.loss = th.nn.CrossEntropyLoss(reduction='none') # TODO: what does reduction do?


    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        seq_length = self.seq_length

        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(
                    x['text'],
                    max_length=seq_length,
                    pad_to_max_length=True)['input_ids']
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset('imdb', split=f'{split}[:{self.batch_size if self.debug else f"{self.percent}%"}]')
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.valid_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'acc': acc}
        print (f"Validation loss {loss}")
        return {**out, 'log': out}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True)

    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.valid_ds,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False)

    def configure_optimizers(self):
        return th.optim.Adam(
                self.model.parameters(),
                lr=self.lr)


def main(*args, **kwargs):
    flags.DEFINE_boolean('debug', False, '')
    flags.DEFINE_integer('epochs', 1, '')
    flags.DEFINE_integer('batch_size', 8, '')
    flags.DEFINE_float('lr', 1e-2, '')
    flags.DEFINE_float('momentum', .9, '')
    flags.DEFINE_string('model', 'bert-base-uncased', '')
    flags.DEFINE_integer('seq_length', 32, '')
    flags.DEFINE_integer('percent', 5, '')

    FLAGS = flags.FLAGS

    # logfile
    sh.rm('-r', '-f', 'logs')
    sh.mkdir('logs')

    params = dict(
            batch_size=FLAGS.batch_size,
            debug=FLAGS.debug,
            percent=FLAGS.percent,
            model=FLAGS.model,
            seq_length=FLAGS.seq_length,
            lr=FLAGS.lr)

    model = IMDBSentimentClassifier(params)
    trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if th.cuda.is_available() else 0),
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            logger=pl.loggers.TensorBoardLogger('logs', name='imdb', version=0)
    )
    trainer.fit(model)

if __name__ == "__main__":
    app.run(main)
