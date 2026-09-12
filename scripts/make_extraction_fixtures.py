"""Generate deterministic extraction fixtures for tests/test_extraction.py.

Run: .\\.venv\\Scripts\\python scripts/make_extraction_fixtures.py
Output: tests/fixtures/extraction/* (committed, stable).

The PDF is hand-built minimal valid PDF (ASCII only) so generation needs
no extra dependencies. The XLSX uses openpyxl (a declared dependency).
"""
from pathlib import Path

OUT = Path(__file__).parent.parent / "tests" / "fixtures" / "extraction"

TXT = """Incident timeline
2026-08-01 09:12 - alert fired on checkout latency
2026-08-01 09:40 - containment: traffic shifted to standby pool
2026-08-01 11:05 - resolved, follow-up scheduled
"""

MD = """# Deployment Runbook

## Prerequisites
- Access to the release checklist
- Change window approved

## Steps
1. Freeze incoming deploys.
2. Roll forward the standby pool.
3. Verify health checks before draining.
"""

CSV = """hostname,status,region
web-01,healthy,us-east-1
web-02,degraded,us-east-1
db-01,healthy,us-east-1
"""

JSON = """{
  "document": "quarterly-report",
  "category": "Financial Reporting",
  "quarter": "Q3",
  "summary": "Revenue grew while expenses stayed flat."
}
"""

HTML = """<html><head><title>Wiki</title></head>
<body>
<h1>Quarterly Planning</h1>
<p>The team agreed on three priorities for the quarter.</p>
<ul><li>Ship the checkout revamp</li><li>Cut standby cost</li></ul>
</body></html>
"""

PDF_LINES = [
    ("Quarterly Incident Review", 20),
    ("Summary", 14),
    ("This quarter saw one major checkout incident.", 11),
    ("Containment", 14),
    ("Traffic was shifted to the standby pool within half an hour.", 11),
    ("Follow-up", 14),
    ("A follow-up review is scheduled with the on-call team.", 11),
]


def _pdf_escape(text):
    return (text.replace("\\", "\\\\").replace("(", "\\(")
            .replace(")", "\\)"))


def build_pdf_bytes():
    ops = []
    y = 750
    for text, size in PDF_LINES:
        ops.append(f"BT /F1 {size} Tf 50 {y} Td "
                   f"({_pdf_escape(text)}) Tj ET")
        y -= 30 if size > 12 else 18
    stream = ("\n".join(ops) + "\n").encode("ascii")
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"),
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii")
        out += body if isinstance(body, bytes) else body.encode("ascii")
        out += b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode("ascii")
    return bytes(out)


def build_xlsx_bytes():
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["Category", "Q3", "Q4"])
    ws.append(["Revenue", 1250000, 1310000])
    ws.append(["Expenses", 820000, 805000])
    ws.append(["Net income", 430000, 505000])
    notes = wb.create_sheet("Notes")
    notes.append(["Note"])
    notes.append(["Figures are synthetic demo data."])
    import io
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ops-notes.txt").write_text(TXT, encoding="utf-8")
    (OUT / "runbook.md").write_text(MD, encoding="utf-8")
    (OUT / "inventory.csv").write_text(CSV, encoding="utf-8")
    (OUT / "config.json").write_text(JSON, encoding="utf-8")
    (OUT / "wiki.html").write_text(HTML, encoding="utf-8")
    (OUT / "incident-review.pdf").write_bytes(build_pdf_bytes())
    (OUT / "budget.xlsx").write_bytes(build_xlsx_bytes())
    for path in sorted(OUT.iterdir()):
        print(f"wrote {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
