from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py")
at.run(timeout=30)
run_btn = [btn for btn in at.button if btn.label == "▶ Run Full SMS Scan"]
run_btn[0].click().run(timeout=120)
print("Errors:")
for e in at.error:
    print(e.value)
