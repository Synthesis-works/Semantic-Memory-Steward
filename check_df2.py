import time
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py")
at.run(timeout=30)
print(at.dataframe[0].value)
