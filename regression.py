import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('csv/input2.csv', header=-1, skiprows=1, encoding='utf-8')

df = df[np.isfinite(df[2])]
df = df[np.isfinite(df[7])]
pop_middle = df[2]
num_away = df[7]

divorce = df[11]
old = df[4]

# plt.plot(old, divorce, 'bo')
# plt.title('regression analysis')
# plt.xlabel('population of old people')
# plt.ylabel('divorce')

plt.plot(pop_middle, num_away, 'bo')
plt.title('regression analysis')
plt.xlabel('population of middle age')
plt.ylabel('number of away from the city')

# model = pd.ols(y=divorce, x=old, intercept=True)
model = pd.ols(y=num_away, x=pop_middle, intercept=True)
print(model)
plt.plot(model.x['x'], model.y_fitted, 'g-')
plt.savefig('data/regression.png')
# plt.savefig('data/regression_old_divorce.png')
plt.show()
