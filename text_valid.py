"""
快速查看验证一些文本
"""


import json
if __name__ == '__main__':
    s = "Huawei(Chile)S. A。地址：Rosario Norte 530, Piso 17Las Condes, Santiago, Chile"
    labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    with open('training.txt', encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            label = sample['标签']
            text = sample['内容']
            if s in text:
                print(text)
                print(label)
                print()
    print('done.')