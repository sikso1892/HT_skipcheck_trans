import re


def sum_processing_time(log_path: str) -> float:
    total = 0.0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"처리시간=([0-9.]+)초", line)
            if m:
                total += float(m.group(1))

    return total


if __name__ == "__main__":
    log_file_path = "/mnt/c/Users/Flitto/Documents/NAC/HT/data/log_20251031_100337.txt"

    total_sec = sum_processing_time(log_file_path)
    print(f"총 처리시간: {total_sec:.2f}초")
