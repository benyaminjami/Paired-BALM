import re
import torch
import anarci


def get_anarci_numbering(sample, max_len=288):
    """
    Calculate anarci numbering and convert anarci numbering to position and chain IDs.

    Args:
        anarci_numbering (list): List of tuples containing the anarci numbering.
        max_len (int, optional): Maximum length of the position IDs. Defaults to 288.

    Returns:
        dict: A dictionary containing the position IDs and chain IDs as torch tensors.
    """
    batch = [('light', sample[0]), ('heavy', sample[1])]
    anarci_numbering = anarci.run_anarci(batch, scheme="imgt")[1]
    anarci_numbering = [numbering[0][0] for numbering in anarci_numbering]

    anarci_numbering_cleaned = []
    for chain_numbering in anarci_numbering:
        numbering = []
        for element in chain_numbering:
            idx = str(element[0][0]) + element[0][1]  # (1, 'A') or (1, ' ')
            aa = element[1]  # 'Q' or '-'
            if aa != "-":
                numbering.append(idx)
        anarci_numbering_cleaned.append(numbering)

    index = []
    chain = []

    indexes = anarci_numbering_cleaned[0]
    c = ([0] * (len(anarci_numbering_cleaned[0]) + 2)) + ([1] * (len(anarci_numbering_cleaned[1]) + 1))
    chain.append(c + ([0] * (max_len - len(c))))
    indexes.append("140")
    indexes.extend(anarci_numbering_cleaned[1])
    try:
        new_index = list(map(lambda id: int(id), indexes))
    except:
        new_index = []
        for id in indexes:
            try:
                new_index.append(int(id))
            except:
                pos_map = {
                    "111A": 129,
                    "111B": 130,
                    "111C": 131,
                    "111D": 132,
                    "111E": 133,
                    "112A": 139,
                    "112B": 138,
                    "112C": 137,
                    "112D": 136,
                    "112E": 135,
                    "112F": 134,
                }
                if (
                    id not in pos_map.keys()
                    and int(re.sub("[a-zA-Z]", "", id)) < 111
                ):
                    new_index.append(int(re.sub("[a-zA-Z]", "", id)))
                elif id in pos_map.keys():
                    new_index.append(pos_map[id])
                elif int(id[:3]) == 111:
                    new_index.append(133)
                elif int(id[:3]) == 112:
                    new_index.append(134)

    new_index = [0] + new_index + [140] * (max_len - 1 - len(new_index))
    if len(new_index) > max_len:
        new_index = new_index[:max_len]
        new_index[-1] = 140
    index.append(new_index)
    return {"position_ids": torch.tensor(index), "chain_ids": torch.tensor(chain)}


if __name__ == "__main__":
    light = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCLQHNSYPRTFGQGTKVEIK"
    heavy = "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDWPFWQWLVRRGERFDYWGQGTLVTVSS"
    print(get_anarci_numbering([light, heavy]))
