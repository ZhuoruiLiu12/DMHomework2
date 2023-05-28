import pandas as pd

# 读取数据
with open('./data/anonymous-msweb.data', 'r') as f:
    lines = f.readlines()

# 提取用户浏览记录
data = []
current_user = None
for line in lines:
    parts = line.strip().split(',')
    if parts[0] == 'C':
        current_user = parts[1]
    elif parts[0] == 'V' and current_user is not None:
        data.append((current_user, parts[1]))

# 转化为 DataFrame
df = pd.DataFrame(data, columns=['user', 'vroot'])

# 使用 get_dummies 方法构造独热编码（one-hot encoded）的数据
basket = df.groupby(['user', 'vroot'])['vroot'].count().unstack().reset_index().fillna(0).set_index('user')
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)
