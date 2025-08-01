import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV to JSON for train/test")
    parser.add_argument("--type", choices=["train", "test"], required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()


def load_csv_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def process_train(rows):
    caption_map = {}
    for row in rows:
        img_name = os.path.basename(row["input_img_path"])
        caption = row["caption"].strip()
        caption_map[img_name] = [caption]
    return caption_map


def process_test(rows):
    return [[os.path.basename(row["input_img_path"]), row["caption"]] for row in rows]


def save_json(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    rows = load_csv_rows(args.csv_path)

    if args.type == "train":
        result = process_train(rows)
    elif args.type == "test":
        result = process_test(rows)
    else:
        raise ValueError(f"Unknown type: {args.type}")

    save_json(result, args.out_json)
    print(
        f"✔ saved {len(result)} {'captions' if args.type == 'train' else 'pairs'} → {args.out_json}"
    )


if __name__ == "__main__":
    main()
