"""Data preparation and preprocessing pipeline for Structured JSON Named Entity Extraction.

Downloads CoNLL-2003, parses token-level BIO tags into structured entity JSON,
and outputs chat-formatted train.jsonl, valid.jsonl, and test.jsonl for MLX fine-tuning.
"""

import io
import json
import logging
import os
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CONLL_URL = "https://data.deepai.org/conll2003.zip"
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert named entity extraction assistant. "
    "Extract all named entities from the text into a valid JSON object matching the following schema exactly:\n"
    "{\n"
    '  "persons": [],\n'
    '  "organizations": [],\n'
    '  "locations": [],\n'
    '  "misc": []\n'
    "}"
)


def download_and_extract_conll(raw_dir: Path) -> Dict[str, Path]:
    """Downloads CoNLL-2003 zip if not present and returns paths to raw split text files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": raw_dir / "train.txt",
        "valid": raw_dir / "valid.txt",
        "test": raw_dir / "test.txt",
    }

    if all(p.exists() for p in splits.values()):
        logger.info("CoNLL-2003 raw files already exist in %s", raw_dir)
        return splits

    logger.info("Downloading CoNLL-2003 dataset from %s...", CONLL_URL)
    req = urllib.request.Request(CONLL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for split_name, file_path in splits.items():
            member_name = f"{split_name}.txt"
            if member_name in z.namelist():
                content = z.read(member_name)
                file_path.write_bytes(content)
                logger.info("Extracted %s (%d bytes)", member_name, len(content))

    return splits


def parse_conll_file(file_path: Path) -> List[Tuple[List[str], List[str]]]:
    """Parses a CoNLL formatted file into a list of (tokens, ner_tags) for each sentence."""
    sentences = []
    curr_tokens = []
    curr_tags = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_tokens:
                    sentences.append((curr_tokens, curr_tags))
                    curr_tokens, curr_tags = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                token = parts[0]
                ner_tag = parts[-1]
                curr_tokens.append(token)
                curr_tags.append(ner_tag)

    if curr_tokens:
        sentences.append((curr_tokens, curr_tags))

    return sentences


def extract_entities(tokens: List[str], tags: List[str]) -> Dict[str, List[str]]:
    """Converts IOB/BIO tags into grouped entity lists according to the project schema."""
    type_map = {
        "PER": "persons",
        "ORG": "organizations",
        "LOC": "locations",
        "MISC": "misc",
    }
    result: Dict[str, List[str]] = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "misc": [],
    }

    current_tokens: List[str] = []
    current_type: str | None = None

    def flush_current():
        nonlocal current_tokens, current_type
        if current_tokens and current_type:
            key = type_map.get(current_type)
            if key:
                entity_str = " ".join(current_tokens)
                result[key].append(entity_str)
        current_tokens = []
        current_type = None

    for token, tag in zip(tokens, tags):
        if tag == "O":
            flush_current()
        elif tag.startswith("B-"):
            flush_current()
            current_type = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-"):
            ent_type = tag[2:]
            if current_tokens and current_type == ent_type:
                current_tokens.append(token)
            else:
                flush_current()
                current_type = ent_type
                current_tokens = [token]
        else:
            flush_current()

    flush_current()
    return result


def reconstruct_sentence(tokens: List[str]) -> str:
    """Joins tokens and normalizes punctuation spacing for natural reading."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.:;?!%'\"]|n't|'s|'m|'re|'ve|'ll|'d)", r"\1", text)
    text = re.sub(r"(\()\s+", r"\1", text)
    text = re.sub(r"\s+(\))", r"\1", text)
    text = re.sub(r"``\s*", '"', text)
    text = re.sub(r"\s*''", '"', text)
    return text


def create_chat_sample(
    user_text: str,
    target_json: Dict[str, Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> Dict[str, Any]:
    """Formats an input-output pair into an MLX chat template format."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(target_json, ensure_ascii=False)},
        ]
    }


def process_and_save_splits(
    raw_splits: Dict[str, Path],
    output_dir: Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> Dict[str, Dict[str, Any]]:
    """Processes all raw CoNLL splits into jsonl chat format and logs statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_stats = {}

    for split_name, raw_path in raw_splits.items():
        sentences = parse_conll_file(raw_path)
        output_file = output_dir / f"{split_name}.jsonl"

        stats = {
            "total_sentences": len(sentences),
            "persons": 0,
            "organizations": 0,
            "locations": 0,
            "misc": 0,
            "sentences_with_zero_entities": 0,
        }

        with open(output_file, "w", encoding="utf-8") as out_f:
            for tokens, tags in sentences:
                user_text = reconstruct_sentence(tokens)
                entities = extract_entities(tokens, tags)

                total_ents = sum(len(v) for v in entities.values())
                if total_ents == 0:
                    stats["sentences_with_zero_entities"] += 1

                for k, v in entities.items():
                    stats[k] += len(v)

                sample = create_chat_sample(user_text, entities, system_prompt)
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        summary_stats[split_name] = stats
        logger.info(
            "Split '%s': %d examples written to %s (PER: %d, ORG: %d, LOC: %d, MISC: %d, 0-ent: %d)",
            split_name,
            stats["total_sentences"],
            output_file,
            stats["persons"],
            stats["organizations"],
            stats["locations"],
            stats["misc"],
            stats["sentences_with_zero_entities"],
        )

    return summary_stats


def main():
    workspace_root = Path(__file__).resolve().parent.parent
    raw_dir = workspace_root / "data" / "raw"
    processed_dir = workspace_root / "data" / "processed"

    print("=" * 60)
    print("  Starting Data Preprocessing Pipeline for CoNLL-2003")
    print("=" * 60)

    raw_splits = download_and_extract_conll(raw_dir)
    stats = process_and_save_splits(raw_splits, processed_dir)

    stats_file = processed_dir / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDataset preparation complete! Summary:")
    for split, stat in stats.items():
        print(f"  [{split.upper()}] {stat['total_sentences']} samples | "
              f"PER: {stat['persons']}, ORG: {stat['organizations']}, "
              f"LOC: {stat['locations']}, MISC: {stat['misc']}")
    print(f"\nStats written to: {stats_file}")


if __name__ == "__main__":
    main()
