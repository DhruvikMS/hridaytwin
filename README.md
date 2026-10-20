# HridayTwin: A Cardiometabolic Digital Twin for Indian Adults

**Happiest Health · Digital Twin Challenge 2026 · Proof-of-Concept submission**

HridayTwin is a research prototype for a personal digital twin that combines clinical records with daily health signals to help explore cardiometabolic risk and personalized care planning for Indian adults with diabetes and/or hypertension.

> **Research and safety notice:** This proof of concept uses synthetic patient data only. It is not a medical device and must not be used to make clinical decisions.

## Team details

| Field | Details |
|---|---|
| Team name | Winmasters |
| Team members | Manjit Kalita, Dhruvik Sorathiya, Bishwarup Roy |
| College | Goa Institute of Management |
| Contact | dhruvik.sorathiya2026b@gim.ac.in |

## Project dashboard

The intended GitHub Pages dashboard URL is:

**https://DhruvikMS.github.io/hridaytwin/dashboard/**

This link will work once the dashboard files are uploaded and GitHub Pages is enabled. The project files still need to be added to this repository before the dashboard can be published.

## Repository contents

The full submission package includes a self-contained dashboard, a Python pipeline, model and validation materials, synthetic sample datasets, documentation, and tests.

## Run locally

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python run_pipeline.py
```

See the documentation in `docs/` for the architecture, data dictionary, validation plan, model card, ethics and privacy notes, and reported results.

## License

See [LICENSE](LICENSE) once the project files have been uploaded.
