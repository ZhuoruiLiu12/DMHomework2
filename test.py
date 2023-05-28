import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# 数据预处理
print("Data Preprocessing...\n")
with open('./data/anonymous-msweb.data') as f:
    content = f.readlines()

# Attribute dictionary for mapping
attr_dict = {}
# Data for storing user history
data = []

# Parsing the file line by line
for line in content:
    line = line.strip().split(',')
    if line[0] == 'A':  # 这是一个属性行，更新字典
        attr_id = int(line[1])
        attr_name = line[3].strip('"')
        attr_dict[attr_id] = attr_name
    if line[0] == 'C':  # 这是一个新的案例（用户）
        data.append(dict())
    if line[0] == 'V':  # 这是一个投票（访问的网站）行，追加到最新的用户
        data[-1][attr_dict[int(line[1])]] = 1

# 将列表的列表转换为DataFrame
df = pd.DataFrame(data).fillna(0)

print("Head of processed data:\n")
print(df.head())  # 展示处理后的数据

# 数据探索性分析
print("Exploratory Data Analysis...\n")
page_visit_counts = df.sum().sort_values(ascending=False)
print("\nTop 10 most visited pages:\n", page_visit_counts.head(10))

plt.figure(figsize=(10,5))
sns.histplot(page_visit_counts, bins=50, kde=False)
plt.title('Distribution of page visit counts')
plt.xlabel('Number of visits')
plt.ylabel('Number of pages')
plt.show()

# 关联规则挖掘
print("\nMining Association Rules...\n")
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)  # 计算频繁项集
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)  # 计算关联规则
rules['antecedents'] = rules['antecedents'].apply(lambda a: ', '.join(list(a)))
rules['consequents'] = rules['consequents'].apply(lambda a: ', '.join(list(a)))

print("\nAssociation rules:\n")
print(rules)

# 可视化关联规则
print("\nVisualizing Association Rules...\n")

sorted_rules = rules[['support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.heatmap(sorted_rules, annot=True, cmap='YlGnBu')
plt.title('Top 10 rules according to lift')

plt.yticks(np.arange(10)+0.5, list(zip(rules.loc[sorted_rules.index, 'antecedents'], rules.loc[sorted_rules.index, 'consequents'])), rotation='horizontal')
plt.show()

# 评估关联规则
print("\nEvaluating Association Rules...\n")
# 计算全置信度
rules['all_confidence'] = rules['support'] / rules[['antecedent support', 'consequent support']].min(axis=1)
# 计算卡方值
rules['chi-square'] = 0
for i, rule in rules.iterrows():
    ct = pd.crosstab(df[rule['antecedents']], df[rule['consequents']])
    _, p, _, _ = chi2_contingency(ct)
    rules.loc[i, 'chi-square'] = p
# 计算 Kulczynski measure
rules['kulczynski'] = 0.5 * (rules['support'] / rules['antecedent support'] + rules['support'] / rules['consequent support'])

print("\nAssociation rules with all evaluation metrics:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'all_confidence', 'chi-square', 'kulczynski']])
