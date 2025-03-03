
import dill

with open("artifacts/preprocessor.pkl", "rb") as file:
    obj = dill.load(file)
    print(type(obj))  # Should be ColumnTransformer
