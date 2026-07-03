# 🤖 AI Trade Engine

> An AI-powered multi-agent trading system built on AutoHedge — LLM-driven trade analysis, a Streamlit dashboard, and Alpaca paper-trading integration.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit) ![Alpaca](https://img.shields.io/badge/Alpaca-Paper%20Trading-brightgreen) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project orchestrates a team of AI agents (via the AutoHedge framework) to analyze equities, generate trade theses, and execute simulated trades:

- **LLM analysis agents** (Groq-hosted models) produce directional theses and risk commentary per ticker
- **Market data** is pulled live via `yfinance`
- **Execution** routes to the Alpaca paper-trading API — no real money involved
- **Streamlit dashboard** (`autohedge_app.py`) drives runs interactively and displays structured outputs
- Run artifacts are logged to `autohedge_runs.jsonl` and `outputs/` for auditability

## 🗂️ Project Structure

```
AI-Trade-Engine/
├── autohedge_app.py       # Streamlit app — agent orchestration + dashboard
├── requirements.txt       # Dependencies
├── agent_workspace/       # Agent state snapshots from sample runs
├── outputs/               # Structured analysis outputs (JSON)
└── autohedge_runs.jsonl   # Run log
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nmadagi/AI-Trade-Engine.git
cd AI-Trade-Engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file (never commit it):

```
GROQ_API_KEY=your_groq_key
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

### 4. Run the dashboard

```bash
streamlit run autohedge_app.py
```

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Agents / LLM | AutoHedge, Groq |
| Market data | yfinance |
| Execution | Alpaca paper-trading API |
| UI | Streamlit |
| Data | Pandas |

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It uses paper trading and AI-generated analysis. Nothing here is financial advice.

## 👤 Author

**Nitin Madagi** | [GitHub](https://github.com/nmadagi) | [Portfolio](https://nmadagi.github.io/portfolio)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
