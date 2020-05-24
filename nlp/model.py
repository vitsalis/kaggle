import sh
import nlp
import random
import transformers
import torch as th
import numpy as np
import pytorch_lightning as pl

class SentimentClassifier(pl.LightningModule):
    def __init__(self, hparams, train_ds, valid_ds):
        super().__init__()

        self.batch_size = hparams['batch_size']
        self.model_name = hparams['model']
        self.lr = hparams['lr']
        self.num_workers = hparams['num_workers']
        self.dropout_pct = hparams['dropout_pct']

        self.train_ds = train_ds
        self.valid_ds = valid_ds

        self.roberta_config = transformers.RobertaConfig.from_pretrained(self.model_name, output_hidden_states=True)
        self.model = transformers.RobertaModel.from_pretrained(self.model_name, config=self.roberta_config)

        self.dropout = th.nn.Dropout(self.dropout_pct)
        self.fc = th.nn.Linear(self.roberta_config.hidden_size, 2)
        th.nn.init.normal_(self.fc.weight, std=0.02)
        th.nn.init.normal_(self.fc.bias, 0)

        #self.model2 = transformers.RobertaForSequenceClassification.from_pretrained(self.model_name)

    def loss_fn(self, start_logits, end_logits, start_positions, end_positions):
        ce_loss = th.nn.CrossEntropyLoss(reduction='none')
        return (ce_loss(start_logits, start_positions) +
                ce_loss(end_logits, end_positions))

    def compute_jacard_score(self, tweets, start_idx, end_idx, start_logits, end_logits, offsets):
        def get_selected_text(tw, start, end, offsets):
            res = ""
            for ix in range(start, end+1):
                res += tw[offsets[ix][0]:offsets[ix][1]]
                if ix + 1 < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
                    # need to add a space
                    res += " "
            return res

        def jacard(x, y):
            a = set(x.lower().split())
            b = set(y.lower().split())
            c = a.intersection(b)
            return float(len(c)) / (len(a) + len(b) - len(c))

        start_preds = np.argmax(start_logits, 1)
        end_preds = np.argmax(end_logits, 1)

        jact = []
        for i, tw in enumerate(tweets):
            if start_preds[i] > end_preds[i]:
                sel = tw
            else:
                sel = get_selected_text(tw, start_preds[i], end_preds[i], offsets[i])

            valid = get_selected_text(tw, start_idx[i], end_idx[i], offsets[i])
            jact.append(jacard(sel, valid))

        print (jact)
        return th.tensor(jact)

    def forward(self, batch):
        _, _, hs = self.model(batch['ids'], batch['masks'])

        # TODO
        # 13 hidden states
        # on the notebook they use only the last 3, I'm gonna use them all
        # and watch for any differences
        out = th.stack(hs)
        out = th.mean(out, 0)
        out = self.dropout(out)
        out = self.fc(out)

        start_logits, end_logits = [x.squeeze(-1) for x in out.split(1, dim=-1)]
        return start_logits, end_logits

    def training_step(self, batch, batch_idx):
        start_logits, end_logits = self.forward(batch)
        loss = self.loss_fn(start_logits, end_logits, batch['start_idx'], batch['end_idx']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        start_logits, end_logits = self.forward(batch)
        loss = self.loss_fn(start_logits, end_logits, batch['start_idx'], batch['end_idx'])

        jac = self.compute_jacard_score(
                batch['tweet'],
                batch['start_idx'],
                batch['end_idx'],
                start_logits,
                end_logits,
                batch['offsets'])
        return {'loss': loss, 'jac': jac}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        jac = th.cat([o['jac'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_jac': jac}
        print ("Loss {:.4f} | Jaccard: {:.4f}".format(loss, jac))
        return {**out, 'log': out}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.valid_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

    def configure_optimizers(self):
        return th.optim.Adam(
                self.model.parameters(),
                lr=self.lr)
