"""Deterministic demo-workspace seeder for SMS dogfooding/demo video.

Builds ~20 synthetic multi-format documents (~1.3 MB total) with varied
size, age (in-content dates; S3 cannot backdate LastModified), category,
and sensitivity, plus one identical duplicate pair and one near-identical
pair for real duplicate detection.

All content is synthetic: fictional org (Northwind Traders), fictitious
names, invented figures. No PII, secrets, or credentials.

Usage:
    .\\.venv\\Scripts\\python scripts/seed_demo_workspace.py [--bucket NAME] [--reset]

- seed uploads exactly the manifest keys (never touches anything else).
- reset deletes exactly the manifest keys (and their memory records when
  a DynamoDB table is provided).
- The application never depends on this script.

Every seeded object stays under the 100 KB reader cap so each document
is genuinely analyzable.
"""
import argparse
import io
import os

ORG = "Northwind Traders"
FICTIONAL = "All names, figures, and events are fictitious demo data."
READER_CAP = 100_000


def _wrap(text, width=88):
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _pad_text(base, filler, target):
    """Deterministically repeat filler blocks until target bytes reached."""
    parts, text = [base], base
    index = 0
    while len(text.encode("utf-8")) < target:
        block = filler(index)
        parts.append(block)
        text = "\n".join(parts)
        index += 1
        if index > 100000:
            break
    return text


def md_doc(title, intro, sections, target):
    base = f"# {title}\n\n*{ORG} - {FICTIONAL}*\n\n{intro}\n"
    blocks = [f"## {heading}\n\n{body}" for heading, body in sections]

    def filler(i):
        return blocks[i % len(blocks)] + f"\n\n_Dated entry {i + 1}._"

    return _pad_text(base, filler, target).encode("utf-8")


def _pdf_escape(text):
    return (text.replace("\\", "\\\\").replace("(", "\\(")
            .replace(")", "\\)"))


def pdf_doc(title, sections, target):
    """Hand-built minimal ASCII PDF (no generation dependency).

    Paginates so documents can reach realistic sizes; every page is
    plain Helvetica text that pypdf extracts deterministically.
    """
    raw_lines = [(title, 20)]
    for heading, body in sections:
        raw_lines.append((heading, 14))
        for paragraph in body:
            raw_lines.extend([(line, 11) for line in _wrap(paragraph)])

    lines, used = [(t, s, False) for t, s in raw_lines], sum(
        len(t) + 10 for t, _ in raw_lines)
    filler_idx = 0
    while used < target:
        lines.append((f"Appendix entry {filler_idx + 1}: status nominal, "
                      "no action required at this time.", 11, True))
        used += 85
        filler_idx += 1
        if filler_idx > 5000:
            break

    def _render(active_lines):
        pages, current, y_pos = [], [], 750
        for text, size, _appendix in active_lines:
            step = 30 if size > 12 else 16
            if y_pos < 60:
                pages.append(current)
                current, y_pos = [], 750
            current.append(f"BT /F1 {size} Tf 50 {y_pos} Td "
                           f"({_pdf_escape(text)}) Tj ET")
            y_pos -= step
        if current:
            pages.append(current)

        objects = ["<< /Type /Catalog /Pages 2 0 R >>", None]
        page_refs, next_num, contents = [], 3, []
        for page_ops in pages:
            stream = ("\n".join(page_ops) + "\n").encode("ascii", "replace")
            contents.append(stream)
            page_refs.append(next_num)
            next_num += 2
        kids = " ".join(f"{num} 0 R" for num in page_refs)
        objects[1] = (f"<< /Type /Pages /Kids [{kids}] "
                      f"/Count {len(pages)} >>")
        font_num = 3 + 2 * len(pages)
        for page_num, stream in zip(page_refs, contents):
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {page_num + 1} 0 R "
                f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>")
            objects.append(
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                + stream + b"endstream")
        objects.append("<< /Type /Font /Subtype /Type1 /BaseFont "
                       "/Helvetica >>")
        assert font_num == len(objects), "object numbering drift"

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

    payload = _render(lines)
    while len(payload) > 96_000:
        appendix = [i for i, line in enumerate(lines) if line[2]]
        if not appendix:
            break
        del lines[appendix[-1]]
        payload = _render(lines)
    assert len(payload) <= READER_CAP
    return payload


def csv_doc(header, row_fn, target):
    lines = [header]
    text = "\n".join(lines) + "\n"
    index = 0
    while len(text.encode("utf-8")) < target:
        lines.append(row_fn(index))
        text = "\n".join(lines) + "\n"
        index += 1
        if index > 100000:
            break
    return text.encode("utf-8")


def json_doc(obj_text, entries_text, target):
    base = "{\n" + obj_text
    parts, text, index = [base], base, 0
    while len(text.encode("utf-8")) < target:
        parts.append('  ' + entries_text.format(i=index))
        text = ",\n".join(parts) + "\n}\n"
        index += 1
        if index > 100000:
            break
    return text.encode("utf-8")


def html_doc(title, heading, paragraphs, list_items, target):
    body = [f"<h1>{heading}</h1>"] + [f"<p>{p}</p>" for p in paragraphs]
    body.append("<ul>" + "".join(f"<li>{item}</li>"
                                 for item in list_items) + "</ul>")
    base = ("<html><head><title>" + title + "</title></head><body>\n"
            + "\n".join(body))
    parts, text, index = [base], base, 0
    while len(text.encode("utf-8")) < target:
        parts.append(f"<h2>Update {index + 1}</h2>\n"
                     f"<p>Status nominal for {ORG.lower()} operations.</p>")
        text = "\n".join(parts) + "\n</body></html>\n"
        index += 1
        if index > 100000:
            break
    return text.encode("utf-8")


def txt_log(header, entry_fn, target):
    return csv_doc(header, entry_fn, target)


def xlsx_workbook(sheets, fixed_props=True):
    """sheets: list of (title, rows) where rows are lists of cell values."""
    from openpyxl import Workbook
    from datetime import datetime
    wb = Workbook()
    first = True
    for title, rows in sheets:
        ws = wb.active if first else wb.create_sheet(title)
        if first:
            ws.title = title
            first = False
        for row in rows:
            ws.append(list(row))
    if fixed_props:
        stamp = datetime(2025, 1, 1, 0, 0, 0)
        wb.properties.created = stamp
        wb.properties.modified = stamp
    buf = io.BytesIO()
    wb.save(buf)
    return _deterministic_zip(buf.getvalue())


def _deterministic_zip(payload):
    """Repack an xlsx (zip) with fixed entry timestamps.

    openpyxl stamps zip entries with the current time and rewrites
    dcterms:modified on every save; both are normalized here so
    identical workbooks produce identical bytes on every run.
    """
    import re
    import zipfile
    fixed_modified = ("<dcterms:modified xsi:type=\"dcterms:W3CDTF\">"
                      "2025-01-01T00:00:00Z</dcterms:modified>")
    with zipfile.ZipFile(io.BytesIO(payload)) as src:
        entries = []
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "docProps/core.xml":
                text = data.decode("utf-8")
                text = re.sub(r"<dcterms:modified[^>]*>[^<]*</dcterms:modified>",
                              fixed_modified, text)
                data = text.encode("utf-8")
            entries.append((info.filename, data, info.compress_type))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dest:
        for filename, data, compress_type in entries:
            info = zipfile.ZipInfo(filename, date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = compress_type
            dest.writestr(info, data)
    return out.getvalue()


def xlsx_sized(sheets_fn, target):
    """Grow row counts until the workbook reaches target bytes.

    Converges from below so the result never exceeds the reader cap.
    Deterministic: fixed content, fixed workbook properties.
    """
    count, best = 50, None
    while True:
        payload = xlsx_workbook(sheets_fn(count))
        if len(payload) > READER_CAP:
            break
        best = payload
        if len(payload) >= target:
            break
        per_row = max(len(payload) / max(count, 1), 1.0)
        count += max(int((target - len(payload)) / per_row * 0.8), 25)
        if count > 200000:
            break
    assert best is not None, "could not build workbook"
    return best


# ---------------------------------------------------------------------------
# Document builders: build_all() -> {key: bytes}
# ---------------------------------------------------------------------------

def _financial_rows(count):
    rows = [[f"2025-Q3 line {i + 1}", "Northwind Traders",
             1000 + i * 37, "approved"] for i in range(count)]
    return [("Ledger", [["Entry", "Org", "Amount", "Status"]] + rows)]


def build_all():
    docs = {}

    docs["demo/architecture-overview.md"] = md_doc(
        "Architecture Overview",
        f"{ORG} checkout platform, current design. {FICTIONAL}",
        [("Services", "Checkout, pricing, and standby pools. Health "
                       "checks gate every drain operation."),
         ("Data", "Orders ledger with nightly reconciliation and "
                   "duplicate detection on idempotency keys.")],
        55_000)

    docs["demo/deployment-runbook.pdf"] = pdf_doc(
        "Deployment Runbook",
        [("Prerequisites", ["Release checklist signed.", "Change window approved."]),
         ("Steps", ["Freeze incoming deploys.", "Roll the standby pool forward.",
                    "Verify health checks before draining."])],
        85_000)

    docs["demo/api-reference.md"] = md_doc(
        "API Reference",
        f"Checkout and pricing endpoints. {FICTIONAL}",
        [("Orders", "Create, fetch, and cancel orders. Idempotency keys "
                    "required on writes."),
         ("Pricing", "Quote endpoint with currency support.")] ,
        65_000)

    docs["demo/incident-postmortem-2025.pdf"] = pdf_doc(
        "Incident Postmortem 2025-11-18",
        [("Timeline", ["09:12 alert fired on checkout latency.",
                       "09:40 containment via standby pool.",
                       "11:05 resolved, follow-up scheduled."]),
         ("Action items", ["Add latency alert.", "Drill standby failover."])],
        80_000)

    near = pdf_doc(
        "Incident Postmortem 2025-11-18 Final",
        [("Timeline", ["09:12 alert fired on checkout latency.",
                       "09:40 containment via standby pool.",
                       "11:05 resolved, follow-up scheduled, notes finalized."]),
         ("Action items", ["Add latency alert.", "Drill standby failover.",
                           "Publish final notes."])],
        80_000)
    docs["demo/incident-postmortem-2025-final.pdf"] = near

    docs["demo/service-config.json"] = json_doc(
        '  "service": "checkout",\n  "org": "Northwind Traders",',
        '  "flag_{i}": "enabled"',
        8_000)

    docs["demo/project-retrospective-2023.md"] = md_doc(
        "Project Retrospective 2023",
        f"Looking back at the 2023 storefront migration. {FICTIONAL}",
        [("Went well", "Nightly reconciliation caught every duplicate."),
         ("Next time", "Freeze deploys earlier in the change window.")],
        55_000)

    docs["demo/q3-financial-review.xlsx"] = xlsx_sized(
        lambda n: [("Review", [["Line", "Org", "Amount", "Note"]] +
                    [[f"2025-Q3 item {i + 1}", ORG, 5000 + i * 113,
                      "Confidential quarterly figures, handle with care."]
                     for i in range(n)])],
        90_000)

    budget = xlsx_sized(
        lambda n: [("Budget", [["Category", "Planned", "Spent"]] +
                    [[f"Cost center {i + 1}", 20000 + i * 410,
                      19000 + i * 390] for i in range(n)])],
        92_000)
    docs["demo/annual-budget.xlsx"] = budget
    docs["demo/annual-budget-final.xlsx"] = budget

    docs["demo/vendor-analysis.pdf"] = pdf_doc(
        "Vendor Analysis 2024",
        [("Method", ["Scored three vendors on cost and support."]),
         ("Outcome", ["Renewed standby pool vendor for 2025."])],
        95_000)

    docs["demo/expense-export.csv"] = csv_doc(
        "date,category,amount,note",
        lambda i: (f"2025-09-{(i % 28) + 1:02d},travel,{120 + i},"
                   f"field visit {i + 1}"),
        70_000)

    docs["demo/milestone-tracker.xlsx"] = xlsx_sized(
        lambda n: [("Milestones", [["Milestone", "Owner", "Status"]] +
                    [[f"Milestone {i + 1}", f"Owner {(i % 5) + 1}",
                      "done" if i % 3 else "planned"] for i in range(n)])],
        88_000)

    docs["demo/project-retrospective-2025.pdf"] = pdf_doc(
        "Project Retrospective 2025",
        [("Wins", ["Standby failover drill passed."]),
         ("Debts", ["Old export jobs still pending cleanup."])],
        92_000)

    docs["demo/research-notes.md"] = md_doc(
        "Research Notes",
        f"Checkout latency experiments. {FICTIONAL}",
        [("Hypothesis", "Standby pools cut tail latency."),
         ("Result", "P99 improved in three of four drills.")],
        50_000)

    docs["demo/system-events.txt"] = txt_log(
        "timestamp,level,service,message",
        lambda i: (f"2024-03-{(i % 28) + 1:02d}T10:00:00Z,INFO,checkout,"
                   f"heartbeat {i + 1} ok"),
        90_000)

    docs["demo/incident-export.csv"] = csv_doc(
        "id,severity,summary",
        lambda i: (f"INC-{(i % 900) + 100},low,"
                   f"routine check {i + 1} closed"),
        60_000)

    docs["demo/operations-dashboard.html"] = html_doc(
        "Operations Dashboard", "Nightly operations overview",
        [f"{ORG} standby pool status nominal.",
         "Duplicate detection running on order idempotency keys."],
        ["Review standby cost", "Confirm drill schedule"],
        15_000)

    docs["demo/team-handbook.pdf"] = pdf_doc(
        "Team Handbook",
        [("People", ["Maya Chen, Tomas Rivera, Priya Nair, Jonas Weber, "
                     "Aisha Bello. All names are fictitious."]),
         ("Norms", ["Write decision logs.", "Review standby cost monthly."])],
        88_000)

    docs["demo/onboarding-notes.md"] = md_doc(
        "Onboarding Notes",
        f"Welcome to {ORG}. {FICTIONAL}",
        [("Week one", "Read the runbook and the decision log."),
         ("Buddy", "Pair with an on-call teammate.")] ,
        40_000)

    for key, payload in docs.items():
        assert len(payload) <= READER_CAP, f"{key} exceeds reader cap"
    return docs


def manifest(docs=None):
    """Stable manifest: key, era, kind. Era is an in-content property
    (S3 cannot backdate LastModified)."""
    if docs is None:
        docs = build_all()
    eras = {
        "demo/architecture-overview.md": "current",
        "demo/deployment-runbook.pdf": "current",
        "demo/api-reference.md": "current",
        "demo/incident-postmortem-2025.pdf": "older",
        "demo/incident-postmortem-2025-final.pdf": "older",
        "demo/service-config.json": "current",
        "demo/project-retrospective-2023.md": "stale",
        "demo/q3-financial-review.xlsx": "current",
        "demo/annual-budget.xlsx": "current",
        "demo/annual-budget-final.xlsx": "current",
        "demo/vendor-analysis.pdf": "older",
        "demo/expense-export.csv": "current",
        "demo/milestone-tracker.xlsx": "current",
        "demo/project-retrospective-2025.pdf": "older",
        "demo/research-notes.md": "current",
        "demo/system-events.txt": "older",
        "demo/incident-export.csv": "older",
        "demo/operations-dashboard.html": "current",
        "demo/team-handbook.pdf": "current",
        "demo/onboarding-notes.md": "current",
    }
    docs = docs if docs is not None else build_all()
    return [{"key": key, "era": eras[key],
             "kind": key.rsplit(".", 1)[-1],
             "bytes": len(docs[key])} for key in sorted(docs)]


def duplicate_pairs():
    """Intentional duplicates: one identical, one near-identical."""
    return [("demo/annual-budget.xlsx", "demo/annual-budget-final.xlsx")]


def seed(s3_client, bucket, prefix="demo/"):
    """Upload exactly the manifest keys. Touches nothing else."""
    docs = build_all()
    keys = []
    for spec in manifest():
        key = spec["key"]
        assert key.startswith(prefix), f"manifest key outside prefix: {key}"
        s3_client.put_object(Bucket=bucket, Key=key, Body=docs[key])
        keys.append(key)
    return keys


def reset(s3_client, bucket, dynamodb_table=None):
    """Delete exactly the manifest keys (and their memory records when a
    table is provided). Never wildcard-deletes."""
    keys = []
    for spec in manifest():
        key = spec["key"]
        s3_client.delete_object(Bucket=bucket, Key=key)
        keys.append(key)
        if dynamodb_table is not None:
            dynamodb_table.delete_item(
                Key={"s3_uri": f"s3://{bucket}/{key}"})
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket",
                        default=os.environ.get("SMS_S3_BUCKET",
                                               "semantic-memory-steward-dev-527557823928"))
    parser.add_argument("--reset", action="store_true",
                        help="delete manifest keys instead of seeding")
    parser.add_argument("--list", action="store_true",
                        help="print manifest and exit")
    args = parser.parse_args()
    if args.list:
        for spec in manifest():
            print(f"{spec['key']} ({spec['kind']}, {spec['era']}, "
                  f"{spec['bytes']} bytes)")
        return
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get(
        "AWS_DEFAULT_REGION", "us-east-1"))
    if args.reset:
        keys = reset(s3, args.bucket)
        print(f"removed {len(keys)} manifest keys")
    else:
        keys = seed(s3, args.bucket)
        total = sum(len(payload) for payload in build_all().values())
        print(f"seeded {len(keys)} documents ({total} bytes)")


if __name__ == "__main__":
    main()
