import csv
import json
import re
import argparse


def normalize(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    s = s.replace('""', '"').replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    return s


def get_source_and_fields(raw: str):
    s = normalize(raw)
    try:
        o = json.loads(s)
        return (
            o.get('sourceAddress'),
            o.get('requestID'),
            o.get('shortMessage'),
        )
    except json.JSONDecodeError:
        # fallback regex extraction
        src = re.search(r'"sourceAddress"\s*:\s*"([^"]*)"', s)
        req = re.search(r'"requestID"\s*:\s*"([^"]*)"', s)
        msg = re.search(r'"shortMessage"\s*:\s*"([^"]*)"', s)
        return (
            src.group(1) if src else None,
            req.group(1) if req else None,
            msg.group(1).replace('\\"', '"') if msg else None,
        )


def extract_messages(
    input_path: str,
    output_path: str,
    originator: str,
    max_rows: int = 1_000_000,
    dilute_factor: int = 10,
):
    """Extract messages from CSV and write filtered rows to a new CSV."""
    with open(input_path, newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["originator_id", "message_id", "raw_text"])
        writer.writeheader()

        for i, row in enumerate(reader, 1):
            src, req, msg = get_source_and_fields(row["Value"])

            if src == originator and i % dilute_factor == 0:
                writer.writerow({
                    "originator_id": src,
                    "message_id": req,
                    "raw_text": msg
                })

            if i % 100_000 == 0:
                print(f"Processed {i:,} rows")

            if i >= max_rows:
                break

    print(f"Done. Output written to: {output_path}")


# ----------------------------
# CLI Support
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract originator messages from CSV.")

    parser.add_argument("--originator", required=True, help="Originator ID to filter")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--max_rows", type=int, default=1_000_000, help="Max number of rows to process")
    parser.add_argument("--dilute", type=int, default=10, help="Take every Nth matching row")

    args = parser.parse_args()

    extract_messages(
        input_path=args.input,
        output_path=args.output,
        originator=args.originator,
        max_rows=args.max_rows,
        dilute_factor=args.dilute
    )
