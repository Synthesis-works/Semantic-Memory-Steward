import time
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py")
at.run(timeout=30)
run_btn = at.button[0]
run_btn.click().run(timeout=120)
print(at.dataframe[0].value)
