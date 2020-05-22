import sh
import nlp
import transformers
import torch as th
import pytorch_lightning as pl

from absl import logging, app, flags

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

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction='none') # TODO: what does reduction do?

        self.train_ds = None
        self.valid_ds = None

    def prepare_data(self):
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
        out = {'loss': loss, 'acc': acc}
        return {**out, 'log': out}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=FLAGS.batch_size,
                shuffle=True,
                drop_last=True)

    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.valid_ds,
                batch_size=FLAGS.batch_size,
                shuffle=False,
                drop_last=False)

    def configure_optimizers(self):
        return th.optim.Adam(
                self.model.parameters(),
                lr=FLAGS.lr)


def main(*args, **kwargs):
    model = IMDBSentimentClassifier()
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
