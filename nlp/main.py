import sh
import torch as th
import pandas as pd
import pytorch_lightning as pl

from absl import logging, app, flags
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from model import SentimentClassifier
from utils import seed_everything
from loader import TweetDataset

def main(*args, **kwargs):
    # TODO: clean these up
    flags.DEFINE_boolean('debug', False, '')
    flags.DEFINE_integer('epochs', 1, '')
    flags.DEFINE_integer('batch_size', 8, '')
    flags.DEFINE_float('lr', 1e-2, '')
    flags.DEFINE_float('momentum', .9, '')
    flags.DEFINE_string('model', 'roberta-base', '')
    flags.DEFINE_integer('seq_length', 32, '')
    flags.DEFINE_integer('percent', 5, '')
    flags.DEFINE_integer('num_workers', 8, '')
    flags.DEFINE_integer('max_length', 96, '')
    flags.DEFINE_integer('seed', 62, '')
    flags.DEFINE_float('split', 0.2, '')

    FLAGS = flags.FLAGS

    seed_everything(FLAGS.seed)
    # logfile
    sh.rm('-r', '-f', 'logs')
    sh.mkdir('logs')

    params = dict(
            batch_size=FLAGS.batch_size,
            debug=FLAGS.debug,
            percent=FLAGS.percent,
            model=FLAGS.model,
            seq_length=FLAGS.seq_length,
            lr=FLAGS.lr,
            split=FLAGS.split,
            num_workers=FLAGS.num_workers)

    ## read input
    data_path = Path("input")
    train_df = pd.read_csv(data_path/"train.csv")
    train_df.dropna(inplace=True)
    test_df = pd.read_csv(data_path/"test.csv")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=FLAGS.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
        train_ds = TweetDataset(train_df.iloc[train_idx], FLAGS.model, FLAGS.max_length)
        val_ds = TweetDataset(train_df.iloc[val_idx], FLAGS.model, FLAGS.max_length)

        model = SentimentClassifier(params, train_ds, val_ds)
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
