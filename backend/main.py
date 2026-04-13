from backend.config import CSV_FILE_PATH
from backend.pipeline.run_pipeline import run_pipeline


def main() -> None:
    summary = run_pipeline(str(CSV_FILE_PATH))
    print("\n========== FINAL SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
