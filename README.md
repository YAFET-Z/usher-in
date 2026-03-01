# 🛡️ Usher-In: GenAI Command Center

**Full-stack observability for Gemini 2.5 Pro.** Turn the AI "black box" into actionable data.

## 🚀 Overview
Usher-In is a high-performance API gateway that provides a real-time "War Room" for LLM operations. Built with **FastAPI** and **Google Vertex AI**, it uses **Datadog** to track cost, latency, and reliability in a single pane of glass.

## 📊 The Command Center
![Usher-In Dashboard]

### Key Metrics Tracked:
* **API Success Rate:** 100% reliability monitoring.
* **P95 Latency:** Identifying bottlenecks in generative content production.
* **Token Burn Rate:** Real-time cost analysis of Gemini 2.5 Pro.

## 🛠️ Tech Stack
* **AI:** Google Gemini 2.5 Pro
* **Backend:** Python / FastAPI
* **Observability:** Datadog APM & Dashboards
* **Instrumentation:** Custom Spans via `ddtrace`

## 🧪 Installation & Setup
1. Clone the repo: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your environment variables in `.env`:
   - `DD_API_KEY`: Your Datadog API Key
   - `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account JSON