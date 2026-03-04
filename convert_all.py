"""Convert Excel files in Data_2026 to CSV files.

Behavior:
- Scans the folder `Data_2026` for .xls/.xlsx/.xlsm files.
- For a workbook with a single sheet, writes `filename.csv`.
- For workbooks with multiple sheets, writes `filename_sheetname.csv` for each sheet.
- Outputs are placed in `Data_2026/csv/`.

Dependencies: pandas, openpyxl (for .xlsx), xlrd (for some .xls). Install with:
    pip install pandas openpyxl xlrd
"""

import os
from pathlib import Path
import pandas as pd

# Configure paths
folder_path = Path("Data_2026")
if not folder_path.exists():
    raise SystemExit(f"Input folder not found: {folder_path.resolve()}")

output_dir = folder_path / "csv"
output_dir.mkdir(parents=True, exist_ok=True)

# Supported Excel extensions
EXTS = {".xls", ".xlsx", ".xlsm"}

files_converted = 0
files_failed = 0

for file in folder_path.iterdir():
    if not file.is_file():
        continue
    if file.suffix.lower() not in EXTS:
        continue
    # Skip temporary/exchange files
    if file.name.startswith("~$"):
        continue

    print(f"Converting {file.name}...")
    try:
        # Use ExcelFile to inspect sheet names (lets pandas pick a suitable engine)
        xls = pd.ExcelFile(file)
        sheet_names = xls.sheet_names

        if len(sheet_names) == 1:
            # Single sheet -> filename.csv
            df = pd.read_excel(xls, sheet_name=sheet_names[0])
            out_path = output_dir / f"{file.stem}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
        else:
            # Multiple sheets -> one CSV per sheet
            for sheet in sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                # Make sheet name filesystem-safe
                safe_sheet = "".join(c if (c.isalnum() or c in (' ', '-', '_')) else '_' for c in sheet).strip()
                out_path = output_dir / f"{file.stem}_{safe_sheet}.csv"
                df.to_csv(out_path, index=False)
                print(f"  Saved: {out_path}")

        files_converted += 1
    except Exception as exc:
        files_failed += 1
        print(f"  Failed to convert {file.name}: {exc}")

print(f"Done. Converted: {files_converted}. Failed: {files_failed}.")