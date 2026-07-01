# clubos-databricks

Production Databricks data platform for ClubOS — Real Madrid's digital analytics OS.

## Structure
- `src/bronze/` — Raw ingestion from Excel source files via Unity Catalog Volumes
- `src/silver/` — Normalization, type casting, unpivot to fact tables
- `src/gold/` — KPI health, peer benchmark, signal validation, priority board
- `src/analytics/` — Pearson correlation signals, priority scoring inputs
- `src/quality/` — Data quality checks across all layers
- `resources/` — Databricks Workflow/Job definition as code
- `seeds/` — metric_dictionary.json (polarity config for 59 metrics)

## Catalog
Unity Catalog: `clubos` → schemas: `bronze`, `silver`, `gold`