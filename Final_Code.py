# Project of Polynomial Regression



# Part 1
import pandas as pd
import matplotlib.pyplot as plt

x = {
    "AdSpend": [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
    "Sales":   [20,45,80,130,200,290,400,540,700,900]
}

df = pd.DataFrame(x)

print(df)

df.to_csv("Advertising.csv", index=False)

plt.plot(df['AdSpend'], df['Sales'])





# Part 2

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np


X = df[['AdSpend']]
y = df['Sales']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Trained Successfully")

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# print(y_test, y_pred)


X_range = np.linspace(1000, 10000, 100).reshape(-1,1)
X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)

plt.scatter(X, y)
plt.plot(X_range, y_range_pred)
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.title("Polynomial Regression Curve (Degree 3)")
plt.show()

spend = 1500
pred = model.predict(poly.transform([[spend]]))
print(f"Predicted Sales for Ad Spend {spend}: {pred[0]:.2f}")

import pickle

pickle.dump(model, open('model.pkl', 'wb'))
# pickle.dump(poly, open('poly.pkl', 'wb'))

print("Model Saved")




# part 3

import gradio as gr
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))


def predict_sales(amount):
    pred = model.predict(poly.transform([[amount]]))[0]
    return f"Predicted Sales: {pred:.2f}"

# UI
app = gr.Interface(
    fn=predict_sales,
    inputs=gr.Number(label="Enter Ad Spend"),
    outputs=gr.Textbox(label="Prediction"),
    title="Ad Spend → Sales Predictor"
)

app.launch(share=True)