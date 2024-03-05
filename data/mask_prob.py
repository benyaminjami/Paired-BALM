import pickle
import torch
import numpy as np
import pandas as pd

if __name__ == "__main__":

    with open('data.pkl', 'rb') as f:
        d = pickle.load(f)

    concatenated_string = ''.join([''.join(inner_list) for inner_list in d['seqs']])

    # Convert the concatenated string into a set of characters
    unique_characters = set(concatenated_string)

    char_to_index_dict = {char: index for index, char in enumerate(unique_characters)}

    heavy_frequencies = np.zeros((140, 20))
    light_frequencies = np.zeros((140, 20))

    for pos, seq in zip(d['pos'], d['seqs']):
        h_pos, l_pos = pos
        h_seq, l_seq = seq
        for p, s in zip(h_pos, h_seq):
            try:
                heavy_frequencies[int(p), char_to_index_dict[s]] += 1
            except:
                pass
        for p, s in zip(l_pos, l_seq):
            try:
                light_frequencies[int(p), char_to_index_dict[s]] += 1
            except:
                pass

    l_freqs = pd.DataFrame(light_frequencies)
    l_freqs = l_freqs[1:129]
    l_probs = l_freqs.divide(l_freqs.sum(1), axis=0)
    l_entropy = ((np.log2(l_probs) * l_probs).fillna(0).sum(1) * -1)
    l_mask = (l_entropy * 0.15 / l_entropy.mean())
    l_mask[l_mask < 0.1] = 0.1
    l_mask.mean()
    new_data = pd.Series([0.2] * 11, index=range(129, 140))
    l_mask = pd.concat([l_mask, new_data])

    h_freqs = pd.DataFrame(heavy_frequencies)
    h_freqs = h_freqs[1:129]
    h_probs = h_freqs.divide(h_freqs.sum(1), axis=0)
    h_entropy = ((np.log2(h_probs) * h_probs).fillna(0).sum(1) * -1)
    h_mask = (h_entropy * 0.15 / h_entropy.mean())
    h_mask[h_mask < 0.1] = 0.1
    h_mask.mean()
    new_data = pd.Series([0.2] * 11, index=range(129, 140))
    h_mask = pd.concat([h_mask, new_data])

    h_mask_tensor = torch.tensor([0] + h_mask.sort_index().values.tolist() + [0])
    l_mask_tensor = torch.tensor([0] + l_mask.sort_index().values.tolist() + [0])
    print(h_mask_tensor)
    print(l_mask_tensor)
    torch.save((h_mask_tensor, l_mask_tensor), 'mask_probs.pt')
