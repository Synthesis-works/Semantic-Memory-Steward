import time
from streamlit.testing.v1 import AppTest

try:
    print("Loading app...")
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    print("Clicking Scan...")
    run_btn = at.button[0] # The first button is the scan button
    run_btn.click().run(timeout=120)
    
    print("Errors during scan:")
    print([getattr(msg, 'value', '') for msg in at.error])
    
    df = at.dataframe[0].value
    print(df)
    
except Exception as e:
    import traceback
    traceback.print_exc()
