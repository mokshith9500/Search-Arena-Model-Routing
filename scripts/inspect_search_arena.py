import json
import random
import re
import os

# ============================================================
# SETTINGS
# ============================================================

DATASET_FILE = "train.jsonl"


# ============================================================
# BASIC HELPERS
# ============================================================

def get_text(content):
    """
    Safely convert message content into readable text.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))

            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def get_messages(record, model):
    """
    Get messages for model_a or model_b.
    """
    key = f"messages_{model}"
    messages = record.get(key, [])

    if not isinstance(messages, list):
        return []

    return messages


def get_user_messages(messages):
    return [
        get_text(m.get("content"))
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    ]


def get_assistant_messages(messages):
    return [
        get_text(m.get("content"))
        for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]


def count_visible_citations(text):
    """
    Count citation markers such as [1], [2], [3].
    """
    if not text:
        return 0

    citations = re.findall(r"\[\d+\]", text)

    return len(citations)


def get_visible_citation_markers(text):
    """
    Return unique citation markers found in the answer.
    """
    if not text:
        return []

    citations = re.findall(r"\[\d+\]", text)

    # Preserve order while removing duplicates
    return list(dict.fromkeys(citations))


def contains_think(text):
    """
    Check whether hidden reasoning markers appear.
    """
    if not text:
        return False

    return "<think>" in text.lower() or "</think>" in text.lower()


# ============================================================
# SEARCH TRACE DISPLAY
# ============================================================

def display_search_trace(trace):
    """
    Display web_search_trace in a readable way.

    We don't assume an exact schema because the dataset's
    trace structure can contain nested objects/lists.
    """

    if trace is None:
        print("  No web_search_trace field.")
        return

    if trace == []:
        print("  web_search_trace = []")
        return

    print(f"  Trace type: {type(trace).__name__}")

    if isinstance(trace, list):
        print(f"  Trace entries: {len(trace)}")

        for i, entry in enumerate(trace, start=1):
            print(f"\n  --- Trace entry {i} ---")

            if isinstance(entry, (dict, list)):
                try:
                    print(
                        json.dumps(
                            entry,
                            indent=2,
                            ensure_ascii=False
                        )
                    )
                except Exception:
                    print(str(entry))
            else:
                print(str(entry))

    elif isinstance(trace, dict):
        print(
            json.dumps(
                trace,
                indent=2,
                ensure_ascii=False
            )
        )

    else:
        print(str(trace))


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

def display_model(record, model):
    """
    Display everything important for one model.
    """

    print("\n" + "=" * 90)
    print(f"MODEL {model.upper()}")
    print("=" * 90)

    messages = get_messages(record, model)

    print(f"\nNumber of messages: {len(messages)}")

    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    user_messages = get_user_messages(messages)

    print("\n" + "-" * 90)
    print("USER MESSAGE(S)")
    print("-" * 90)

    if not user_messages:
        print("No user message found.")

    else:
        for i, message in enumerate(user_messages, start=1):
            print(f"\n[User message {i}]\n")
            print(message)

    # --------------------------------------------------------
    # ASSISTANT ANSWERS
    # --------------------------------------------------------

    assistant_messages = get_assistant_messages(messages)

    print("\n" + "-" * 90)
    print("ASSISTANT MESSAGE(S)")
    print("-" * 90)

    if not assistant_messages:
        print("No assistant message found.")

    else:

        for i, answer in enumerate(assistant_messages, start=1):

            print(f"\n[Assistant turn {i}]")
            print("-" * 50)

            print(answer)

            citation_markers = get_visible_citation_markers(answer)

            print("\nVisible citation markers:")
            print(citation_markers if citation_markers else "None")

            print(
                f"Visible citation count: "
                f"{len(citation_markers)}"
            )

            if contains_think(answer):
                print("WARNING: <think> block detected.")

    # --------------------------------------------------------
    # SEARCH TRACE
    # --------------------------------------------------------

    trace = record.get(
        f"system_{model}_metadata",
        {}
    )

    print("\n" + "-" * 90)
    print("SYSTEM METADATA")
    print("-" * 90)

    if isinstance(trace, dict):

        print("\nMetadata keys:")
        print(list(trace.keys()))

        search_trace = trace.get("web_search_trace")

        print("\n" + "=" * 70)
        print("WEB SEARCH TRACE")
        print("=" * 70)

        display_search_trace(search_trace)

    else:
        print("No system metadata found.")

    # --------------------------------------------------------
    # MODEL CONFIG
    # --------------------------------------------------------
    # llm_config is nested inside system_{model}_metadata, not a
    # top-level "llm_config_{model}" key (verified against real
    # records — the top-level key never exists).

    metadata = record.get(f"system_{model}_metadata", {})

    config = (
        metadata.get("llm_config")
        if isinstance(metadata, dict)
        else None
    )

    if config is not None:

        print("\n" + "-" * 90)
        print("LLM CONFIG")
        print("-" * 90)

        try:
            print(
                json.dumps(
                    config,
                    indent=2,
                    ensure_ascii=False
                )
            )
        except Exception:
            print(config)


# ============================================================
# LOAD ONE RECORD
# ============================================================

def get_record(record_number):
    """
    Read exactly one record from the JSONL file.

    record_number is 1-based.
    """

    if record_number < 1:
        return None

    try:

        with open(
            DATASET_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            for current_number, line in enumerate(f, start=1):

                if current_number == record_number:

                    line = line.strip()

                    if not line:
                        return None

                    return json.loads(line)

    except FileNotFoundError:

        print(
            f"\nERROR: Could not find {DATASET_FILE}"
        )

        print(
            "\nMake sure this Python file is in the "
            "same folder as train.jsonl."
        )

        return None

    except json.JSONDecodeError as e:

        print(
            f"\nERROR: Could not parse record "
            f"{record_number}."
        )

        print(e)

        return None

    return None


# ============================================================
# COUNT RECORDS
# ============================================================

def count_records():
    """
    Count records without loading the dataset.
    """

    print("\nCounting records...")
    print("(This may take a little while for the 1.2 GB file.)")

    count = 0

    try:

        with open(
            DATASET_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if line.strip():
                    count += 1

        print(f"\nTotal records: {count}")

        return count

    except FileNotFoundError:

        print(
            f"\nERROR: Could not find {DATASET_FILE}"
        )

        return 0


# ============================================================
# INSPECT RECORD
# ============================================================

def inspect_record(record_number):

    print("\n")
    print("=" * 100)
    print(f"INSPECTING RECORD #{record_number}")
    print("=" * 100)

    record = get_record(record_number)

    if record is None:

        print(
            "\nCould not find that record."
        )

        return

    # --------------------------------------------------------
    # GENERAL RECORD INFORMATION
    # --------------------------------------------------------

    print("\n" + "-" * 90)
    print("GENERAL INFORMATION")
    print("-" * 90)

    print(f"\nRecord number: {record_number}")

    print(
        f"Winner: "
        f"{record.get('winner', 'NOT FOUND')}"
    )

    print(
        f"Primary intent: "
        f"{record.get('primary_intent', 'NOT FOUND')}"
    )

    print(
        f"Secondary intent: "
        f"{record.get('secondary_intent', 'NOT FOUND')}"
    )

    print(
        f"Languages: "
        f"{record.get('languages', 'NOT FOUND')}"
    )

    print(
        f"Judge: "
        f"{record.get('judge', 'NOT FOUND')}"
    )

    # --------------------------------------------------------
    # MODEL IDENTITIES
    # --------------------------------------------------------

    print("\n" + "-" * 90)
    print("MODEL IDENTITIES")
    print("-" * 90)

    print(
        "\nModel A:",
        record.get("model_a", "NOT FOUND")
    )

    print(
        "Model B:",
        record.get("model_b", "NOT FOUND")
    )

    # --------------------------------------------------------
    # DISPLAY BOTH MODELS
    # --------------------------------------------------------

    display_model(record, "a")

    display_model(record, "b")

    # --------------------------------------------------------
    # RAW RECORD KEYS
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("TOP-LEVEL RECORD KEYS")
    print("=" * 90)

    print(
        json.dumps(
            list(record.keys()),
            indent=2
        )
    )

    print("\n" + "=" * 100)
    print("END OF RECORD")
    print("=" * 100)


# ============================================================
# RANDOM RECORD
# ============================================================

def inspect_random_record():

    total = count_records()

    if total == 0:
        return

    number = random.randint(1, total)

    print(
        f"\nRandomly selected record: #{number}"
    )

    inspect_record(number)


# ============================================================
# MAIN MENU
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("SEARCH ARENA 24K — MANUAL RECORD INSPECTOR")
    print("=" * 80)

    print(
        f"\nDataset file: {DATASET_FILE}"
    )

    if os.path.exists(DATASET_FILE):

        size_gb = (
            os.path.getsize(DATASET_FILE)
            / (1024 ** 3)
        )

        print(
            f"Dataset size: {size_gb:.2f} GB"
        )

    else:

        print(
            "\nWARNING: train.jsonl was not found."
        )

    while True:

        print("\n")
        print("=" * 80)

        print("""
Choose an option:

1. Inspect a specific record
2. Inspect a random record
3. Count total records
4. Exit
""")

        choice = input("Choice: ").strip()

        # ----------------------------------------------------
        # OPTION 1
        # ----------------------------------------------------

        if choice == "1":

            value = input(
                "\nEnter record number "
                "(example: 100): "
            ).strip()

            try:

                record_number = int(value)

                inspect_record(
                    record_number
                )

            except ValueError:

                print(
                    "\nPlease enter a valid number."
                )

        # ----------------------------------------------------
        # OPTION 2
        # ----------------------------------------------------

        elif choice == "2":

            inspect_random_record()

        # ----------------------------------------------------
        # OPTION 3
        # ----------------------------------------------------

        elif choice == "3":

            count_records()

        # ----------------------------------------------------
        # OPTION 4
        # ----------------------------------------------------

        elif choice == "4":

            print(
                "\nExiting. Goodbye!"
            )

            break

        else:

            print(
                "\nInvalid choice. "
                "Please enter 1, 2, 3, or 4."
            )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()