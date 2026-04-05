import csv
from pathlib import Path
from typing import Dict, List

from utils.logger import get_logger

logger = get_logger("LOADER")

CSV_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "iso-8859-1", "cp1252", "utf-16")


def _normalize_row_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {str(k).replace("\ufeff", "").strip(): (v or "") for k, v in row.items()}


def _build_issue_text(row: Dict[str, str]) -> str:
    topic = row.get("Topic", "").strip()
    subtopic = row.get("SubTopic", "").strip()
    subject = row.get("Subject", "").strip()
    description = row.get("Description", "").strip()
    troubleshooting = row.get("Troubleshooting StepsTaken", "").strip()
    product = row.get("Product", "").strip()
    severity = row.get("Severity Level", "").strip()

    parts = [
        f"product: {product}" if product else "",
        f"severity: {severity}" if severity else "",
        f"topic: {topic}" if topic else "",
        f"subtopic: {subtopic}" if subtopic else "",
        f"subject: {subject}" if subject else "",
        f"description: {description}" if description else "",
        f"troubleshooting: {troubleshooting}" if troubleshooting else "",
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
                            "resolution": row.get("Resolution", "").strip(),
                            "next_steps": row.get("Next Steps", "").strip(),
                        }
                    )

            logger.info("Loaded CSV using encoding: %s", encoding)
            logger.info("Total records loaded: %d", len(records))
            if records:
                logger.debug("Sample issue records: %s", records[:2])
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