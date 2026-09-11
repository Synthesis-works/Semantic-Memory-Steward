# SMS Engineering Handover

==================================================
1. PROJECT IDENTITY
==================================================

Project:
Semantic Memory Steward (SMS)

GitHub:
Synthesis-works/Semantic-Memory-Steward

Purpose:
SMS is a semantic storage governance agent for professional/team cloud storage.

Core idea:
Observe S3 objects 
-> understand their meaning 
-> enrich semantic evidence 
-> score importance 
-> identify relationships/duplicates 
-> apply deterministic governance policy 
-> persist semantic memory 
-> authorize actions 
-> request human approval when required 
-> execute safe S3 mutations 
-> verify the resulting state

Core architectural principle:
- LLM determines what a document means.
- Deterministic systems determine what should happen.
- Authorization determines whether an action is permitted.
- Human approval is required for sensitive/high-risk decisions.
- ActionEngine is the only mutation boundary.

==================================================
2. HACKATHON CONTEXT
==================================================

SMS is being developed for the Agents for Humans Hackathon.

Important confirmed rules:
- Strands Agents SDK is required.
- Bedrock AgentCore is encouraged but NOT required.
- Five judging categories are equally weighted: Technical Implementation, Design, Potential Impact, Creativity & Originality, Presentation.
- Public GitHub repository is required.
- Architecture diagram required.
- Demo video maximum 5 minutes.
- MIT or Apache license required.

Do not claim AgentCore is implemented unless the repository proves it.
Do not claim Bedrock inference works unless live evidence proves it.

==================================================
3. CURRENT GIT STATE
==================================================

- Current branch: feature/agentic-ux
- Latest commit: 0b304ad (fix: lowercase recommended_action to prevent pydantic Literal ValidationErrors)

IMPORTANT:
feature/agentic-ux contains the current unmerged dashboard/UX work.
main contains the previously hardened product at a4dba77.
Do not assume feature/agentic-ux is already merged into main.
- Working tree: Clean
- Most recent relevant commit hashes:
  - 0b304ad (Pydantic Literal ValidationError fix)
  - c2d49ed (UI input for fallback API key)
  - a4dba77 (fix: implement ARCHIVE, duplicate safety, and ActionAuthorizer blocks)
- Whether hardening was merged to main: Yes, a4dba77 was merged to main.

Known recent hardening commit: a4dba77
It implemented/finalized:
- ARCHIVE
- duplicate safety
- ActionAuthorizer protections
- related hardening tests

Current reported full test count: 164/164 passing

==================================================
4. COMPLETE ARCHITECTURE
==================================================

Expected conceptual path:
S3 -> S3ContentReader -> SMSPipeline -> Strands Agent -> semantic analysis -> AWS Comprehend enrichment -> importance scoring -> relationship analysis -> PolicyEngine -> semantic memory -> ActionAuthorizer -> ActionEngine -> S3 mutation -> verification

- Strands handles semantic reasoning (LLM).
- Comprehend provides AWS semantic/entity/PII enrichment (Deterministic/API).
- Importance scoring is deterministic.
- Relationship detection is deterministic.
- Policy is deterministic.
- Authorization is deterministic.
- S3 actions are deterministic and verified.

==================================================
5. STRANDS / LLM PROVIDERS
==================================================

Strands:
1.54.0

Bedrock model:
amazon.nova-micro-v1:0

SageMaker provider:
available in installed Strands version

Primary agent:
src/sms_agent/agent.py
- Configured default provider/model: Driven by SMS_LLM_PROVIDER environment variable, currently mocked or forced to gemini in UI.
- Structured output model: Uses Pydantic SemanticAnalysisResult.
- Provider configuration: Configured in src/sms_agent/agent.py.
- External fallback implementation: Fallback logic for Groq, NVIDIA, Mistral, Gemini is implemented manually in agent.py by calling REST endpoints when standard Strands abstractions do not work.
- How provider failure is represented: Hard exceptions if keys are missing, handled upstream in app.py.
- Exact environment variables involved: SMS_LLM_PROVIDER, SMS_EXTERNAL_API_KEY, SMS_EXTERNAL_MODEL.

CRITICAL: AWS Bedrock currently has an account-level restriction.
Known historical behavior: First controlled Strands inference attempt returned account verification/access denial. Later controlled retry returned ValidationException: Operation not allowed.
No retry loop should exist.
Do not claim Bedrock works.
External fallback is currently used for development/demo purposes.
The UI exposes provider status visually in the sidebar, accurately describing the fallback state and AWS quotas.

==================================================
6. AWS BEDROCK SUPPORT CASE
==================================================

Account: 527557823928
Region: us-east-1
Support case: 178868592800270

Known issue: Bedrock model invocation is blocked at the account level.
Known responses include: AccessDeniedException: Your account is currently being verified. and later ValidationException: Operation not allowed.
Support case was previously Unassigned.

Do NOT invent its current status.
Do NOT repeatedly retry Bedrock.
Do NOT close the case.
This is an AWS account/access issue, not currently proven to be an SMS application bug.

==================================================
7. AWS RESOURCES
==================================================

S3 application bucket:
- semantic-memory-steward-dev-527557823928
- Region: us-east-1
- Demo objects: demo/employee-contacts.txt, demo/financial-report.txt, demo/old-project-log.txt, demo/project-plan.txt.
- Controlled quarantine test involving: trash/demo/employee-contacts-copy.txt.

DynamoDB:
- sms-semantic-memory

S3 Vectors:
- vector bucket: sms-semantic-vectors-527557823928
- index: sms-embeddings
- Configuration: dimension = 768, data type = float32, distance = cosine

CloudFormation/SAM stack:
- sms-semantic-memory-infrastructure
- Infrastructure components include: DynamoDB table, S3 Vectors provisioning Lambda, Lambda layer containing boto3, custom resource, IAM role.

Important:
- The existing application S3 bucket is NOT created/managed by the infrastructure stack.
- DynamoDB has Retain behavior.
- S3 Vectors delete behavior was hardened to retain underlying vectors.
- Do not create new infrastructure unless explicitly requested.

==================================================
8. REAL AWS VALIDATION ALREADY COMPLETED
==================================================

Real S3 ingestion validation successfully:
- discovered actual objects
- read them through S3ContentReader
- computed SHA-256 hashes
- importance scoring worked
- relationships worked
- policy evaluation worked
- objects persisted to DynamoDB
- read-back worked
- idempotency was verified
- no S3 PUT/DELETE occurred during the ingestion validation
- no Bedrock/Gemini inference was used during that validation

S3 Vectors/DynamoDB persistence smoke test succeeded using a synthetic 768-dimensional vector.
Comprehend live validation succeeded against employee contacts.
Comprehend detected PERSON entities. It did NOT detect PII.

Important architectural lesson: PERSON entity != PII.
Do not introduce any logic that automatically treats PERSON as PII.

==================================================
9. SAGEMAKER INVESTIGATION
==================================================

Strands has a native SageMaker provider.
A temporary investigation attempted: meta-textgeneration-llama-3-8b-instruct with ml.g5.2xlarge.
AWS returned: ResourceLimitExceeded.
The account-level service quota for ml.g5.2xlarge for endpoint usage is 0.

Temporary SageMaker model/configuration resources were cleaned up. Temporary IAM role was cleaned up. No endpoint was left running. No real SageMaker inference occurred.

DO NOT retry/deploy SageMaker unless explicitly requested.

==================================================
10. POLICY ENGINE
==================================================

Known intended behavior:
- high/medium-risk sensitive documents -> REVIEW
- destructive actions -> human approval
- hard delete remains strongly protected
- HIGH importance + LOW risk -> KEEP
- LOW importance + LOW risk + stale -> ARCHIVE
- LOW importance + LOW risk + not stale -> KEEP
- duplicate evidence -> REVIEW

The duplicate relationship must be passed into PolicyEngine. This was previously a real bug and has been fixed.

==================================================
11. AUTHORIZATION
==================================================

Important intended invariant:
- LOW-risk ARCHIVE may be autonomously authorized when policy permits it.
- AUTONOMOUS QUARANTINE should remain blocked.
- High-risk mutations must not bypass human approval.
- DELETE remains protected.
- Human approval cannot override policy prohibitions.
Do not weaken these protections.

==================================================
12. ACTIONENGINE
==================================================

ARCHIVE: copy -> verify destination -> delete source -> verify source removed
QUARANTINE: follows the same safety pattern.
The system must never report VERIFIED unless the resulting S3 state was actually verified.

Known regression tests include:
- successful archive
- delete failure
- source remains after supposed delete
- authorization behavior

==================================================
13. SEMANTIC MEMORY
==================================================

DynamoDB persists the SemanticMemoryRecord structure including hash, score, and decision. S3 Vectors stores the embedding. Idempotency is implemented. Currently used primarily as persistence infrastructure and for duplicate detection, rather than a fully exploited active knowledge graph.

==================================================
14. EVALUATION HARNESS
==================================================

Files: evaluation/fixtures.py, evaluation/runner.py.
Cases include: project plan, old project log, employee contacts, financial report, duplicate financial report.
Offline evaluation mocks the LLM result.
Live mode uses the real pipeline/Strands and fails honestly when Bedrock is unavailable.
Never represent offline evaluation as live model performance.

==================================================
15. CURRENT DASHBOARD
==================================================

Known UI sections: SMS Status, Run Full SMS Scan, document table, Document Inspector, Semantic Evidence, AWS Comprehend Enrichment, Approval & Action.
Current dashboard status: Functional but NOT considered finished. UX is problematic.

Observed issues:
1. It feels more like a technical admin/data dashboard than an agentic product.
2. KEEP + Confirm & Execute appeared to do nothing visibly.
   The KEEP issue is an observed user-facing failure, not a known root cause.
   Do not assume it is merely a rendering problem.
   Reproduce: REVIEW document -> KEEP -> Confirm & Execute.
   Then trace the actual callback/action/persistence/session-state path.
   The fix must make the real state transition work, not merely display a success message.
3. Button interactions do not consistently communicate state transitions.
4. Action results are not always clearly reflected in the UI.
5. The user cannot always tell what happened after clicking an action.
6. The dashboard needs stronger reasoning/explanation.
7. The dashboard should better communicate what SMS found, why SMS made a decision, what SMS is about to do, what happened, whether the user needs to act.
8. Shows provider/infrastructure status too prominently (like a debugging console).
9. Product needs to feel more like an autonomous semantic steward and less like a CRUD interface.

THIS IS THE CURRENT PRIMARY DEVELOPMENT AREA.

==================================================
16. PRODUCT UX DIRECTION
==================================================

The next development phase should NOT add random backend features. The goal is to make the existing backend feel like a real agent.

Desired conceptual experience:
SMS analyzes workspace
SMS summarizes what it found
SMS identifies safe actions
SMS autonomously handles low-risk permitted cleanup
SMS pauses when human judgment is required
Human approves/rejects
ActionEngine executes
SMS reports verified outcome

Workspace summary required: Documents analyzed, Needs attention, Safe to keep, Safe cleanup candidates, Duplicates. All calculated dynamically.
Recommendations should explicitly show "Why" in bullet points.
Do NOT hardcode the demo filenames into runtime behavior.

==================================================
AGENTICITY DEFINITION
==================================================

SMS should feel agentic because it:
1. observes a workspace
2. reasons about what it finds
3. forms recommendations
4. distinguishes autonomous vs human-required actions
5. acts when authorized
6. pauses when human judgment is required
7. verifies actions
8. reports outcomes

Do NOT create fake:
- "thinking" animations
- simulated activity logs
- fabricated autonomous events
- artificial agent messages

Expose real backend decisions and real action lifecycle.

==================================================
17. ACTION FEEDBACK REQUIREMENTS
==================================================

Every user action must produce an obvious state transition. Never "click -> nothing".
For KEEP: human selects KEEP -> confirmation -> authorization -> no S3 mutation -> result -> UI refresh -> explicit "Kept" state.
For QUARANTINE: human approval -> authorization -> executing -> copying -> destination verified -> source deleted -> source removal verified -> success.
For ARCHIVE: policy permits -> authorization -> ActionEngine -> archive -> verification -> UI reflects archived state.
For FAILED: show action attempted, failure, reason, known source/destination state, next step where appropriate. Never fake success.

==================================================
18. STREAMLIT STATE
==================================================

Streamlit reruns after button clicks.
Ensure: selected document persists, selected action persists appropriately, results survive reruns, scan results survive reruns, action feedback survives reruns, stale state is not shown as current state, UI session state does not become a second persistence database.
Persistent business state belongs in backend storage. Session state is for UI state only.

==================================================
19. HARDCODE AUDIT
==================================================

A search of src/ for employee-contacts.txt, financial-report.txt, old-project-log.txt, project-plan.txt confirms they are NOT driving production behavior. They exist only in tests, fixtures, and documentation. The audit currently passes.

==================================================
20. TEST STATUS
==================================================

Expected command: .\.venv\Scripts\pytest -v
Exact final result: 164 passed in 19.14s (164/164 tests passing).
Includes: UI tests, policy tests, authorization tests, action tests, pipeline tests, Strands tests, evaluation tests.

==================================================
21. README / LICENSE / SUBMISSION
==================================================

LICENSE contains the MIT License.
README.md documents the project architecture.
docs/ folder contains Devpost guidelines.

==================================================
22. KNOWN PRODUCT LIMITATIONS
==================================================

- AWS Bedrock account restriction.
- SageMaker quota unavailable.
- External provider currently needed for development/demo.
- Dashboard UX still needs hardening.
- Provider status needs polished presentation.
- Some semantic-memory capabilities may be persistence scaffolding rather than fully exploited retrieval.

==================================================
23. DO NOT REPEAT THESE INVESTIGATIONS
==================================================

Do NOT waste time repeating:
- SageMaker g5.2xlarge deployment
- repeated Bedrock retries
- basic S3 ingestion validation
- S3 Vectors existence validation
- basic DynamoDB persistence smoke test
- Comprehend PERSON-vs-PII investigation
- duplicate-policy bug investigation
- ARCHIVE implementation investigation

==================================================
24. CURRENT PRIORITY
==================================================

The next priority is: POLISH THE DASHBOARD INTO A REAL AGENTIC PRODUCT EXPERIENCE.
Not new infrastructure, models, or backend features.
First reproduce the dashboard problems, then fix them using TDD.

==================================================
25. RECOMMENDED NEXT WORK
==================================================

P0:
- fix KEEP action feedback/state
- fix all action result visibility
- fix Streamlit rerun/session state problems
- make scan visibly transition through states
- make policy/reasoning understandable

P1:
- workspace summary
- agent recommendations
- clear "needs your attention" queue
- clearer action consequences
- clearer verified outcomes
- polished provider status

P2 only if time permits:
- activity/history
- richer semantic-memory visualization
- additional polish

Do NOT expand scope before P0 is working.

==================================================
DEFINITION OF DONE
==================================================

The dashboard is ready when I can personally perform these flows without confusion:

1. Run Full SMS Scan
2. See what SMS found
3. Understand why each document received its policy
4. See which actions SMS can perform autonomously
5. See which documents require human approval
6. Select REVIEW document
7. Choose KEEP
8. Confirm
9. See explicit successful KEEP result
10. Choose QUARANTINE
11. See execution and verification
12. See explicit final state
13. Observe ARCHIVE recommendation/action
14. See failures honestly
15. Refresh/rerun Streamlit
16. Confirm state is not misleading

If those flows work and the full test suite passes, STOP.
Do not continue adding features.

==================================================
26. HANDOVER PRINCIPLE
==================================================

SMS is NOT currently "unfinished because Bedrock is unavailable."
The core product and safety architecture are substantially implemented.
The current challenge is making the product experience match the quality of the backend.
Target: "An intelligent storage steward with a human safety boundary."
Not: "An S3 document classifier dashboard."

==================================================
27. FINAL HANDOVER SUMMARY
==================================================

CURRENT STATUS:
Backend fully tested (164/164) and hardened. AWS infrastructure exists. Action safety boundaries are secure. UI/UX currently feels like a database viewer rather than an intelligent agent, missing state transitions for actions like KEEP.

LAST VERIFIED TEST RESULT:
164 passed in 19.14s

CURRENT BRANCH:
feature/agentic-ux

LATEST COMMIT:
0b304addaadaf93be4d1c930f239d91898375ccc

AWS BLOCKERS:
Bedrock account-level validation exception (Operation not allowed). SageMaker ml.g5.2xlarge endpoint usage quota is 0.

CURRENT PRIMARY TASK:
Polish the dashboard/product UX into a real agentic experience without touching backend safety logic.

DO NOT:
- deploy SageMaker
- retry Bedrock repeatedly
- add new AWS services
- hardcode demo files into UI
- weaken PolicyEngine, ActionAuthorizer, or ActionEngine
- bypass human approval
- perform unnecessary destructive AWS operations
- merge or push without explicit review

PRODUCTION CODE MAY BE MODIFIED:
- app.py
- UI/session-state code
- relevant tests
- small backend fixes only if a real UI integration bug requires them

Do not modify the established safety architecture unless a real defect is discovered and proven with a regression test.

NEXT AGENT FIRST STEP:
Trace the exact Streamlit execution path in app.py when KEEP is selected to understand why it produces no visible state transition, then TDD the fix.
