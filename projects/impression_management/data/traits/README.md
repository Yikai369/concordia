# Audience traits spreadsheet format

Place your spreadsheet in this folder. For your current experiment, use:

- `projects/impression_management/data/traits/autism-measures-compilation.xlsx`

## Expected structure

- Row 1 is the header row.
- Each column header is treated as a trait `name` (e.g., a survey/category label).
- Each non-empty cell under a column is treated as a trait `assertion`.

The parser in `concordia/prefabs/entity/impression_management_audience.py` reads all columns and all non-empty rows.
