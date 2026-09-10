# Semantic Memory Steward (SMS) - Demo Scenario

## The Concept

SMS turns passive cloud storage into a governed semantic memory engine. 

Traditional storage systems know a file's name, size, and creation date, but they have no idea what the file *means* or what business risks it contains. SMS uses deterministic infrastructure combined with LLM analysis to understand the meaning of documents and safely orchestrate governance decisions, like flagging sensitive data, finding duplicates, and retaining important context.

**Crucially, SMS is a strict steward: the AI can understand a document, but it cannot overrule the governance layer.**

## The Demo Flow (3-5 minutes)

### 1. Introduce the Problem
- Show a typical enterprise S3 bucket that has accumulated files over time.
- Highlight the problem: "We have gigabytes of unstructured data. Some of it is important project documentation. Some is stale history. Some might be highly sensitive employee or financial data. A traditional storage admin just sees files and bytes."

### 2. Show the Target Documents
Show the four synthetic test documents in the demo S3 bucket:
1. project-plan.txt
2. old-project-log.txt
3. employee-contacts.txt
4. inancial-report.txt

*(Optional: we will also show how a duplicate of the financial report behaves)*

### 3. Run SMS 
Execute the SMS pipeline. Explain that SMS is inventorying the bucket, hashing the files, reading the contents, and using AI to parse semantic meaning. 

### 4. Semantic Understanding
Show the output of the analysis:
- project-plan.txt -> High importance, Internal category.
- old-project-log.txt -> Low importance, stale History.
- employee-contacts.txt -> Confidential sensitivity. 
- inancial-report.txt -> Restricted sensitivity.

### 5. Deterministic Governance (The "Killer Moment")
Emphasize the deterministic policy engine routing the files based on the semantic understanding:
- **Project Plan:** LOW RISK + HIGH IMPORTANCE → **KEEP**.
- **Old Project Log:** LOW IMPORTANCE + STALE → **ARCHIVE**.
- **Employee Contacts & Financial Report:** SENSITIVE / HIGH RISK → **REVIEW**.
  
*Call out:* "Notice what happened here. The system didn't just guess what to do. Because it detected restricted material, the PolicyEngine explicitly forced the documents into a REVIEW state, halting any autonomous workflow. The AI provides the insight; the deterministic engine enforces the safety."

### 6. Destructive Action Safety
Demonstrate that a destructive action (like deleting a sensitive document) is impossible to perform autonomously.
- *Explain:* "Even if a hallucinating model recommended delete for our inancial-report.txt, the SMS PolicyEngine hard-blocks deletion of sensitive files and always falls back to requiring human approval."

### 7. The Duplicate Relationship Case
Introduce the duplicate file duplicate-report.txt (a copy of the financial report).
- Show how the deterministic hash catches the exact duplicate.
- Show how SMS flags the relationship and surfaces it for REVIEW rather than blindly deleting it. 

### 8. Persistence (Semantic Memory)
Explain that SMS is not just a script—it's building a memory. 
- Show the DynamoDB table populated with semantic metadata.
- Mention the S3 Vector Memory architecture storing embeddings.
- *Call out:* "Because SMS remembers what it learned, future operations are faster and cheaper. It doesn't need to re-read the financial report every time; it already knows it's highly sensitive."

### 9. Conclusion
"SMS turns cloud storage from passive files into governed semantic memory."

---

## Live Demo Checklist After Bedrock Access

Once AWS Support clears the Bedrock access restriction, follow this sequence to run the demo live:

1. **Verify Bedrock Status:** Run python check_bedrock.py to ensure 
ova-micro-v1:0 is accessible without throwing a ValidationException.
2. **Clear Previous State (Optional):** Empty the DynamoDB table and S3 vectors bucket to show a completely fresh run.
3. **Execute Live Evaluation:**
   `ash
   # Run the live evaluation harness against the S3 bucket using the Strands integration
   PYTHONPATH=src python evaluation/runner.py --live
   `
4. **Inspect Persistence:** Open the AWS Console (DynamoDB) and show the persisted memory records.
