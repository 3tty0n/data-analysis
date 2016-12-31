import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.ensemble as ske
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/csv/input.csv')
del(df['name'])
reg = ske.RandomForestClassifier()
death_elder = df[['death', 'elder']]
reg.fit(df, death_elder)

fet_ind = np.argsort(reg.feature_importances_)[::-1]
fet_imp = reg.feature_importances_[fet_ind]

ax = plt.subplot(111)
plt.bar(np.arange(len(fet_imp)),
        fet_imp, width=1, lw=2)
plt.grid(False)
ax.set_xticks(np.arange(len(fet_imp))+.5)
ax.set_xticklabels(df.head(0))
plt.xlim(0, len(fet_imp))
plt.show()
