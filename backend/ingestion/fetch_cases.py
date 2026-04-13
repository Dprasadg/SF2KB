import csv
from pathlib import Path
from typing import Dict, List

from utils.logger import get_logger

logger = get_logger("LOADER")

CSV_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "iso-8859-1", "cp1252", "utf-16")


def _normalize_row_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {str(k).replace("\ufeff", "").strip(): (v or "") for k, v in row.items()}


def _first_non_empty(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_issue_text(row: Dict[str, str]) -> str:
    issue = _first_non_empty(row, "Issue", "Subject")
    topic = _first_non_empty(row, "Topic")
    subtopic = _first_non_empty(row, "SubTopic")
    subject = _first_non_empty(row, "Subject")
    environment = _first_non_empty(row, "Environment")
    description = _first_non_empty(row, "Description")
    symptoms = _first_non_empty(row, "Symptoms")
    troubleshooting = _first_non_empty(
        row,
        "Troubleshooting StepsTaken",
        "Troubleshooting steps",
        "Troubleshooting Steps",
    )
    product = _first_non_empty(row, "Product")
    severity = _first_non_empty(row, "Severity Level", "Severity")
    root_cause = _first_non_empty(row, "Root Cause", "Cause")

    parts = [
        f"issue: {issue}" if issue else "",
        f"product: {product}" if product else "",
        f"severity: {severity}" if severity else "",
        f"environment: {environment}" if environment else "",
        f"topic: {topic}" if topic else "",
        f"subtopic: {subtopic}" if subtopic else "",
        f"subject: {subject}" if subject else "",
        f"description: {description}" if description else "",
        f"symptoms: {symptoms}" if symptoms else "",
        f"troubleshooting: {troubleshooting}" if troubleshooting else "",
        f"root cause observed: {root_cause}" if root_cause else "",
    ]
    return " | ".join([p for p in parts if p])


def load_case_records_from_csv(file_path: str) -> List[Dict[str, str]]:
    csv_path = Path(file_path)
    logger.info("Loading CSV from: %s", csv_path)

    for encoding in CSV_ENCODING_CANDIDATES:
        records: List[Dict[str, str]] = []
        try:
            with csv_path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                logger.info("CSV Columns (%s): %s", encoding, reader.fieldnames)

                for raw_row in reader:
                    row = _normalize_row_keys(raw_row)
                    issue_text = _build_issue_text(row)
                    if not issue_text:
                        continue

                    records.append(
                        {
                            "issue_text": issue_text,
                            "resolution": _first_non_empty(row, "Resolution"),
                            "next_steps": _first_non_empty(row, "Next Steps"),
                            "troubleshooting": _first_non_empty(
                                row,
                                "Troubleshooting StepsTaken",
                                "Troubleshooting steps",
                                "Troubleshooting Steps",
                            ),
                            "root_cause": _first_non_empty(row, "Root Cause", "Cause"),
                        }
                    )

            logger.info("Loaded CSV using encoding: %s", encoding)
            logger.info("Total records loaded: %d", len(records))
            return records
        except UnicodeDecodeError:
            logger.warning("Failed decoding CSV with encoding: %s", encoding)
            continue
        except Exception:
            logger.exception("Error loading CSV: %s", csv_path)
            raise

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        (
            "Unable to decode CSV using supported encodings: "
            f"{', '.join(CSV_ENCODING_CANDIDATES)}"
        ),
    )


def load_cases_from_csv(file_path: str) -> List[str]:
    # Backward-compatible API for callers expecting only issue text strings.
    return [record["issue_text"] for record in load_case_records_from_csv(file_path)]
