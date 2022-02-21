from typing import List, Dict
from itertools import groupby



def negative_samping(rows: List[Dict]) -> list:
    """负采样

    Args:
        rows (List[Dict]): 同一个用户的展现&点击列表

    Returns:
        list: 负采样结果
    """
    res = []
    if not rows:
        return rows
    # 同一用户按pos正排
    rows = sorted(rows, key=lambda x: x['pos'], reverse=False)
    logid_group = groupby(rows, key=lambda x: x["logid"])
    res = []
    for key, group in logid_group:
        group = list(group)
        keep = False
        click_min = 99999
        click_max = -1
        for i, row in enumerate(group):
            if row['label'] == 1:
                keep = True
                if i < click_min:
                    click_min = i
                if i > click_max:
                    click_max = i
        if not keep:
            continue
        keep_min = max(0, click_min-5)
        keep_max = click_max + 1
        res.extend(group[keep_min:keep_max])
    return res

def rows_flat_map(rows: List[Dict]) -> list:
    res = []
    for row in rows:
        res.append((row['feat_ids'], row['feat_vals'], row['label']))
    return res

def build_data():
    data = [
        {
            'logid': '111',
            'pos': 0,
            'feat_ids': '1,2,3',
            'feat_vals': '4,5,6',
            'label': 0
        },
        {
            'logid': '111',
            'pos': 1,
            'feat_ids': '1,4,6',
            'feat_vals': '2,5,6',
            'label': 1
        },
        {
            'logid': '222',
            'pos': 0,
            'feat_ids': '3,4,7',
            'feat_vals': '4,5,6',
            'label': 1
        },
        {
            'logid': '222',
            'pos': 1,
            'feat_ids': '5,6,7',
            'feat_vals': '8,9,9',
            'label': 0
        }
    ]
    return data


if __name__ == '__main__':
    data = build_data()
    print('data:', data)
    sample = negative_samping(data)
    print('sample', sample)
