import time
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py")
at.run(timeout=30)
print([e.value for e in at.error])
