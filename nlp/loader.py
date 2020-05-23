import transformers
import torch as th

class TweetDataset(th.utils.data.Dataset):
    def __init__(self, df, model_name, max_len=96):
        self.df = df
        self.max_len = max_len
        self.labeled = 'selected_text' in df
        self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(model_name)
        self.sentiment_ids = dict(neutral=0, positive=1, negative=2)

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)

        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        # add space padding before text
        tweet = " " + row.text.lower()
        encoding = self.tokenizer.encode_plus(
                tweet,
                return_offsets_mapping=True)
        sentiment_id = self.tokenizer.encode_plus(row.sentiment).input_ids

        # TODO: why 2??
        ids = [0] +  sentiment_id + [2, 2] + encoding.input_ids + [2]
        offsets = [(0, 0)] * 4 + encoding.offset_mapping + [(0, 0)]

        # TODO: why these specific numbers??
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = th.tensor(ids)
        offsets = th.tensor(offsets)
        # TODO: Do we really need a mask??
        masks = th.where(ids != 1, th.tensor(1), th.tensor(0))

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        """Return the start and end position of the selected sentence"""

        selected_text = row.selected_text.lower()

        idx0 = tweet.find(selected_text)
        idx1 = idx0 + len(selected_text)

        char_targets = th.zeros(len(tweet))
        char_targets[idx0:idx1+1] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        return target_idx[0], target_idx[-1]
