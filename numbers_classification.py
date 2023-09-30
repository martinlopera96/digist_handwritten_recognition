import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

digits = load_digits()
print(digits.data[0])

image = np.reshape(digits.data[10], (8, 8))
print(image)

plt.imshow(image, cmap='gray')

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)

logistic_reg = LogisticRegression().fit(x_train, y_train)

y_pred = logistic_reg.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, linewidths=0.5, square=True, cmap='coolwarm')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
