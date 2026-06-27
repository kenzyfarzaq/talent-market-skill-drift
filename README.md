<div align="center">

# Talent Market Skill Drift Study

### Quantifying how data talent demand shifted across 10 job roles in the US market using Temporal Frequency Analysis, PMI-weighted co-occurrence networks, and TF-IDF adapted role profiling on 785K+ job postings.

<br>

[![License](https://img.shields.io/badge/license-MIT-3DA639?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](notebooks/)
[![Dataset](https://img.shields.io/badge/dataset-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/lukebarousse/data_jobs)

</div>

## 📝 Table of Contents

1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset](#-dataset)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Getting Started](#-getting-started)
7. [Usage](#-usage)
8. [Screenshots](#-screenshots)
9. [Results and Performance](#-results-and-performance)
10. [Limitations](#-limitations)
11. [License](#-license)
12. [Contact](#-contact)
---

## 📌 Overview

This project is a reproducible market intelligence study that analyzes the temporal dynamics of skill demand in the US data job market throughout 2023. Working from a dataset of `785,741` real job postings spanning `10` distinct data roles, the analysis identifies which skills gained momentum, which declined, how skills cluster into co-occurrence ecosystems, and what competency profiles distinguish each role from others. The study is structured as a three-layer analytical framework: temporal proportion tracking (TFA), PMI-weighted co-occurrence network analysis, and TF-IDF adapted distinctiveness scoring. The primary output is a set of `15` analytical visualizations and an auto-generated synthesis of key findings, designed to support evidence-based career planning, hiring strategy, and curriculum design decisions.

---

## ❓ Problem Statement

**Context:** The data job market experienced significant turbulence in 2023. Following the public release of large language model tools in late 2022, demand signals for AI-adjacent skills began shifting visibly. However, most publicly available skill analyses focus on point-in-time frequency counts and fail to distinguish between skills that are merely popular and skills that are actively gaining or losing momentum within a defined period.

**Gap:** There was no publicly available, reproducible analysis that combined temporal momentum detection with skill ecosystem mapping and role-specific distinctiveness profiling on the same dataset. Frequency counts alone cannot answer whether SQL is becoming more or less central, which skills bridge different technology communities, or what truly differentiates a Data Engineer from a Machine Learning Engineer beyond surface-level job titles.

**Solution:** This study applies three complementary analytical methods to a full-year dataset of `784,140` unique job postings: (1) normalized temporal proportion tracking with Chi-Square significance testing to detect genuine skill shifts, (2) PMI-filtered co-occurrence network analysis with Louvain community detection to map ecosystem structure, and (3) TF-IDF adapted scoring to surface role-specific skill distinctiveness beyond raw frequency.

---

## 📊 Dataset

### Metadata

| Attribute | Detail |
|---|---|
| **Source** | Hugging Face Hub |
| **Identifier** | [`lukebarousse/data_jobs`](https://huggingface.co/datasets/lukebarousse/data_jobs) |
| **Raw Size** | `785,741` rows × `17` columns |
| **After Preprocessing** | `784,140` rows (deduplication: 0.21% removed) |
| **Format** | Parquet (loaded via HF `datasets` API) |
| **Geographic Scope** | Global; US-dominant (`26.3%` of postings from United States) |
| **Temporal Coverage** | 2023-01-01 to 2023-12-31 — all 12 months valid (`>= 3,000` postings each) |

### Data Dictionary (Key Columns)

| Column | Type | Description | Example |
|---|:---:|---|---|
| `job_title_short` | `str` | Standardized role category (10 unique values) | `"Data Analyst"` |
| `job_posted_date` | `datetime` | Posting timestamp | `"2023-01-15 10:23:04"` |
| `job_skills` | `str` | Stringified list of required skills (raw) | `"['python', 'sql', 'aws']"` |
| `job_type_skills` | `str` | Stringified dict of skills by category | `"{'cloud': ['aws', 'azure'], ...}"` |
| `job_country` | `str` | Posting country | `"United States"` |
| `job_work_from_home` | `bool` | Remote work indicator | `True` |
| `salary_year_avg` | `float` | Annual salary in USD (97.2% missing) | `95000.0` |

> **Note on `salary_year_avg`:** `97.2%` of values are missing. Companies typically do not disclose salary ranges publicly in job postings. Salary-related analysis in this study is supplementary and does not represent full market compensation data.

### Reproducing the Data Load

```python
from datasets import load_dataset

ds = load_dataset("lukebarousse/data_jobs")
df_raw = ds['train'].to_pandas()

# Shape: (785741, 17)
# All subsequent preprocessing steps are handled in the notebook (Section 5)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role in Project |
|:---:|:---:|---|
| Language | Python 3.10+ | Core analytical pipeline |
| Environment | Jupyter Notebook | Interactive development and reproducible output |
| Data Acquisition | Hugging Face `datasets` | Remote loading of `lukebarousse/data_jobs` |
| Data Processing | Pandas, NumPy | DataFrame operations, normalization, matrix construction |
| Statistical Testing | SciPy (`chi2_contingency`) | Significance testing for skill emergence/decline |
| Network Analysis | NetworkX + python-louvain | Graph construction, PMI weighting, Louvain community detection, centrality metrics |
| Visualization | Matplotlib + Seaborn | 15 analytical figures |
| Version Control | Git + GitHub | Repository management |

---

## 📁 Project Structure

<details>
<summary>Repository structure</summary>
<br>

```
talent-market-skill-drift/
│
├── notebooks/
│   └── talent_market_skill_drift_study.ipynb   ← START HERE
│       (13 sections, fully self-contained and reproducible end-to-end)
│
├── outputs/
│   └── figures/                     # 15 PNG visualizations generated by the notebook
│       ├── plot_bridge_skills.png            # [Sec 9] Bridge skills betweenness analysis
│       ├── plot_category_trend.png           # [Sec 7] Trend by skill category
│       ├── plot_geo_distribution.png         # [Sec 6] Top 15 countries by posting volume
│       ├── plot_missing_values.png           # [Sec 4] Missing value profile per column
│       ├── plot_role_distribution.png        # [Sec 6] Job postings by role
│       ├── plot_role_overlap.png             # [Sec 10] Role cosine similarity matrix
│       ├── plot_role_over_time.png           # [Sec 6] Role proportion stacked area chart
│       ├── plot_skill_count_dist.png         # [Sec 6] Skill count distribution per posting
│       ├── plot_skill_diverging.png          # [Sec 8] Emerging vs. declining skills (H1 vs H2)
│       ├── plot_skill_network.png            # [Sec 9] Co-occurrence network (Kamada-Kawai)
│       ├── plot_skill_opportunity_matrix.png # [Sec 8] Growth rate × frequency bubble chart
│       ├── plot_skill_role_heatmap.png       # [Sec 10] TF-IDF distinctiveness heatmap
│       ├── plot_skill_trend_line.png         # [Sec 7] Monthly skill proportion trends (top 10)
│       ├── plot_temporal_coverage.png        # [Sec 3] Monthly posting volume gate validation
│       └── plot_top_skills_overall.png       # [Sec 6] Top 30 skills overall
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

</details>

> **Entry point:** Open `notebooks/talent_market_skill_drift_study.ipynb` and run all cells sequentially. No pre-downloaded data or external files are required. The dataset is fetched automatically from Hugging Face.

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Minimum Version | Verify |
|---|:---:|---|
| Python | 3.10+ | `python --version` |
| pip | 23.0+ | `pip --version` |
| Git | 2.30+ | `git --version` |
| Jupyter Notebook / Lab | any recent | `jupyter --version` |

### Installation

1. Clone the repository
```bash
git clone https://github.com/kenzyfarzq/talent-market-skill-drift.git
cd talent-market-skill-drift
```

2. Create and activate a virtual environment
```bash
python -m venv venv

source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows (CMD)
# venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

3. Install all dependencies

```bash
pip install -r requirements.txt
```

### Configuration

No additional configuration required. The dataset (`lukebarousse/data_jobs`) is loaded directly from Hugging Face as a public dataset. An HF token is not required, though setting one via `HF_TOKEN` environment variable will enable higher download rate limits.

```bash
# Optional — for faster dataset download
export HF_TOKEN=your_token_here   # macOS / Linux
```

---

## ▶️ Usage

### Running the Notebook

```bash
# Launch Jupyter from the project root
jupyter notebook

# Open the notebook:
# notebooks/talent_market_skill_drift_study.ipynb
```

Run all cells sequentially from top to bottom. The notebook is divided into 13 self-contained sections:

| Section | Title | Description |
|:---:|---|---|
| 1 | Import Library | All dependencies with graceful fallback for optional packages |
| 2 | Reproducibility Config | Centralized parameters (edit here to adjust analysis scope) |
| 3 | Data Acquisition | Load dataset, temporal coverage validation gate |
| 4 | Data Quality Assessment | Missing value profile, duplicate check, skill parsing test |
| 5 | Preprocessing Pipeline | Filter, dedup, parse skills, feature engineering |
| 6 | Exploratory Data Analysis | Role, geography, skill landscape, skill count distribution |
| 7 | Temporal Skill Demand Analysis | Monthly proportion matrix, trend lines, category trends |
| 8 | Skill Emergence and Decline | H1 vs H2 Chi-Square test, diverging bar chart, opportunity bubble chart |
| 9 | Skill Co-occurrence Network | PMI construction, Louvain clustering, bridge skill analysis |
| 10 | Role-Based Skill Profiling | TF-IDF adapted distinctiveness, role overlap matrix |
| 11 | Strategic Synthesis | Auto-generated findings summary |
| 12 | Kesimpulan | Written analytical conclusions |
| 13 | Catatan and Temuan Utama | Analytical caveats and key observations |

### Adjusting Analysis Parameters

All configurable parameters are defined in **Section 2** of the notebook:

```python
RANDOM_STATE       = 42      # Global random seed
MIN_SKILL_FREQ     = 500     # Minimum total frequency for inclusion in growth analysis
TOP_N_SKILLS_EDA   = 30      # Skills shown in EDA overview chart
TOP_N_SKILLS_NET   = 75      # Skills included in co-occurrence network
TOP_N_SKILLS_TREND = 10      # Skills shown in temporal trend line chart
TOP_N_SKILLS_MAT   = 40      # Skills included in TF-IDF role matrix
MIN_ROLE_SIZE      = 1000    # Minimum postings for a role to be included
MIN_MONTHLY_VOL    = 3000    # Monthly volume gate for temporal validity
PMI_THRESHOLD      = 0.3     # Minimum PMI score for a co-occurrence edge
EDGE_WEIGHT_MIN    = 50      # Minimum co-occurrence count for an edge
DATE_START         = '2023-01-01'
DATE_END           = '2023-12-31'
```

---

## 🎥 Screenshots

### Skill Opportunity Matrix

<p align="center">
  <img width="700" height="550" alt="image"
       src="https://github.com/user-attachments/assets/1fb1ae55-5d14-40f7-95d3-89d24a7e155e" />
</p>

Growth momentum (x-axis) vs. overall market demand (y-axis) across all statistically significant skills. Four quadrants: Core & Growing, Emerging, Established Decline, Niche & Fading.

### Skill Emergence vs Decline

<p align="center">
  <img width="500" height="600" alt="image"
       src="https://github.com/user-attachments/assets/593a71e2-6935-4edf-b27f-e6c3928295cc" />
</p>

Top 20 emerging and top 20 declining skills by H1 vs. H2 growth rate (%, p < 0.05). Only skills with total frequency >= 500 are included.

### Co-occurrence Network

<p align="center">
  <img width="700" height="550" alt="image"
       src="https://github.com/user-attachments/assets/3adbf5ca-9b5b-44c3-81e1-3def97fb7add" />
</p>

Top 75 skills, 1,125 edges, PMI threshold >= 0.3. Node size = degree centrality. Color = Louvain community (3 clusters). Layout: Kamada-Kawai.

---

## 📈 Results and Performance

### Key Findings

1. **SQL and Python are now prerequisites, not differentiators.** SQL appeared in `49.0%` (`384,084` mentions) and Python in `48.5%` (`380,158` mentions) of all postings, making both skills present in nearly half of the entire market. TF-IDF analysis confirms both have low distinctiveness across roles precisely because of this ubiquity.

2. **Hugging Face showed the highest statistically significant growth in H2 2023 (+78.2%, p < 0.0001)**, followed by Ubuntu (+43.9%), Unity (+21.1%), and Microsoft Teams (+20.5%). This pattern reflects expanding LLM adoption and broader AI tooling integration into data workflows during 2023.

3. **Platforms and legacy tools declined sharply.** DataRobot (-44.5%), Watson (-34.0%), Vue (-29.4%), and Theano (-26.3%) all showed statistically significant drops in H2, indicating structural market displacement rather than seasonal fluctuation. Of `162` qualified skills tested, `95` declined significantly versus only `17` that rose.

4. **The data skill ecosystem splits into 3 distinct communities.** Louvain community detection on a 75-node PMI network (`>= 0.3`) identified: (1) a DevOps/Engineering cluster (PostgreSQL, Git, Kubernetes, Docker), (2) an ML/Python stack cluster (PyTorch, C++, Pandas, scikit-learn), and (3) an Analytics/BI tools cluster (VBA, Oracle, SPSS, Tableau).

5. **Bridge skills carry disproportionate strategic value.** GitLab and Jupyter show high betweenness centrality while maintaining mid-level degree centrality, meaning they connect multiple technology communities without being universal. These are the highest-value skills for cross-ecosystem career mobility.

### Summary Table

| Finding | Metric |
|---|:---:|
| Total postings analyzed | `784,140` |
| Unique skills tracked | `252` |
| Skills significantly rising (H1 → H2) | `17` |
| Skills significantly declining (H1 → H2) | `95` |
| Top emerging skill (by growth rate) | Hugging Face (`+78.2%`) |
| Top declining skill (by growth rate) | DataRobot (`-44.5%`) |
| Skill communities detected | `3` |
| Most connected skill (degree centrality) | MongoDB |
| Top bridge skill (betweenness centrality) | Ruby* |
| Most similar role pair | Data Scientist — Senior Data Scientist (`1.00`) |
| Most differentiated role pair | Business Analyst — ML Engineer (`0.44`) |

*See Notes section on Ruby's centrality interpretation.

### Skill Trend Lines

<p align="center">
  <img width="700" height="315" alt="image"
       src="https://github.com/user-attachments/assets/62d2332d-fb5e-4aee-9bef-62956ecdff35" />
</p>

Monthly proportion (%) of the top 10 skills. Y-axis uses normalized proportions instead of raw counts to control for seasonal fluctuations in posting volume.

### Role Distribution Over Time

<p align="center">
  <img width="700" height="330" alt="image"
       src="https://github.com/user-attachments/assets/5a801d7a-34ae-46a3-b2d6-76f7995b546f" />
</p>

Stacked area chart showing the share of each role in total postings per month. Stable proportions suggest relatively constant hiring mix throughout 2023.

---

## ⚠️ Limitations

- **Ruby betweenness centrality anomaly.** Ruby has the highest betweenness centrality (`0.5546`) in the co-occurrence network, which likely reflects hybrid job postings that combine data engineering requirements with backend software development. This positioning may not represent Ruby as a genuine bridge skill within a pure data analytics ecosystem. GitLab and Jupyter are more reliable bridge skill candidates with better interpretive grounding.

- **Single-year scope.** The analysis covers 2023 only. It cannot distinguish structural market shifts from cyclical or one-time events (e.g., post-layoff hiring freezes, post-ChatGPT demand spike). Longitudinal comparison with 2022 or 2024 data would strengthen trend interpretations.

- **PMI edge interpretation.** Skill pairs with very high PMI but low co-occurrence count (below `EDGE_WEIGHT_MIN = 50`) are filtered out. Remaining edges with high PMI and low absolute count should be read as "niche co-occurrence," not universal skill pairings.

---

## 📄 License

Distributed under the **MIT License**.
See [LICENSE](LICENSE) for full details.

---

## 📬 Contact

**Ahmad Kenzy Farzaq**

[![GitHub](https://img.shields.io/badge/GitHub-kenzyfarzq-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/kenzyfarzq)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kenzyfarzq-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/kenzyfarzq-60b790320/)
[![Email](https://img.shields.io/badge/Email-kenzyfarzq@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:kenzyfarzq@gmail.com)

<br>

> 💬 Found a bug or have a suggestion? [Open a new issue](https://github.com/kenzyfarzq/workforce-scheduling-optimization/issues/new).

---

<div align="center">
  <sub>
    ⭐ If you find this project useful, consider giving it a star!
    <br>
    Made by <a href="https://github.com/kenzyfarzq">kenzyfarzq</a> · 2026-06
  </sub>
</div>
