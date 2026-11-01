<div align="center">



<p>✦ &nbsp; ✧ &nbsp; ✦</p>



<h1>ARCHIVUM</h1>



<p><strong>Multi-Channel RAG for Warhammer 40,000</strong></p>



<p>✦ &nbsp; ✧ &nbsp; ✦</p>



</div>



Archivum is a local Warhammer 40,000 chat assistant built with retrieval-augmented generation (RAG). It draws on two sources: the *Core Rules* PDF booklet and a SQLite datasheet database.  
Its answers identify the evidence used. 





<div align="center">



<h2>✦ SETUP ✦</h2>



<p><em>Enter the archive</em></p>



</div>



Run the following commands from the repository root. Use Python 3.11 or newer. Create a Python virtual environment:



```bash

python -m venv .venv

```



Activate it using the command for your operating system:



```bash

# macOS or Linux

source .venv/bin/activate

```



```powershell

# Windows PowerShell

.venv\Scripts\Activate.ps1

```



Install the dependencies:



```bash

python -m pip install -r requirements.txt

```



Start the local server:



```bash

python http_adapter.py

```



<div align="center">



<h2>✦ OPEN THE APPLICATION ✦</h2>



<p><em>Open the archive</em></p>



</div>



Visit **http://127.0.0.1:8000** in a browser. Enter your Gemini API key in the interface, then start chatting.



The local server keeps the key and conversation history in memory for the session. It does not write the key to a project file.



> **First-run note:** The first PDF question may take longer because the embedding model must be downloaded and document embeddings computed. This requires an internet connection.





<div align="center">



<h2>✦ EXAMPLE QUESTIONS ✦</h2>



<p><em>Explore the rules and records</em></p>



</div>



| Source | Question |
| --- | --- |
| **Core Rules** | “What happens when a unit Advances?” |
| **Datasheets** | “What is Abaddon’s Toughness?” |
| **Both sources** | “What result is needed to wound Abaddon with a Strength 10 weapon?” |
| **Conversation** | After asking about Abaddon, ask “What weapons does he have?” |



PDF evidence uses identifiers such as `[P1]`; SQL evidence uses identifiers such as `[S1]`. Check the cited evidence when an exact rule or statistic matters.



<div align="center">



<h2>✦ REPOSITORY GUIDE ✦</h2>



<p><em>Inside the archive</em></p>



</div>



| Path | Contents |
| --- | --- |
| `data/processed/core_rules_clean.md` | Parsed and corrected rulebook, included for reproducibility. |
| `data/processed/chunks/structured_chunks.jsonl` | Prepared chunks used for PDF retrieval. |
| `data/raw/wahadb.sqlite` | SQLite datasheet database. |
| `src/` | Backend, routing, retrieval, SQL, parsing, and chunking modules. |
| `dist/` | Browser interface. |
| `http_adapter.py` | Local server connecting the interface to the backend. |
| `evaluation/` | Benchmark, evaluation scripts, and saved results. |



The main Python entry point is `ask_rag(question, history=None, api_key=None)` in `src/rag_backend.py`. The browser interface accesses it through `http_adapter.py`.



For the system design, experimental comparisons, results, and limitations, see the technical report.
