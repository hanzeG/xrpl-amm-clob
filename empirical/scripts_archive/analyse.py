import json
from pathlib import Path

from pyspark.sql import SparkSession

# Path to Delta Sharing profile (repo-root relative)
REPO_ROOT = Path(__file__).resolve().parents[2]
profile_path = REPO_ROOT / "data" / "config.share"

# All tables discovered from discovery.py output
tables = [
    "ripple-ubri-share.ripplex.fact_amm_funding",
    "ripple-ubri-share.ripplex.fact_amm_fees",
    "ripple-ubri-share.ripplex.fact_amm_creates",
    "ripple-ubri-share.ripplex.fact_amm_swaps",
    "ripple-ubri-share.ripplex.offers_fact_tx",
    "ripple-ubri-share.ripplex.fact_amm_bids",
    "ripple-ubri-share.ripplex.fact_amm_votes",
]

spark = SparkSession.builder.getOrCreate()

all_schemas = {}

# Directory to save sample rows
sample_dir = REPO_ROOT / "data" / "sample_rows"
sample_dir.mkdir(parents=True, exist_ok=True)

for full_name in tables:
    table_url = f"{profile_path}#{full_name}"
    try:
        df = spark.read.format("deltaSharing").load(table_url)

        # Collect schema fields
        fields = []
        for f in df.schema.fields:
            fields.append({
                "name": f.name,
                "type": f.dataType.simpleString(),
                "nullable": f.nullable
            })
        all_schemas[full_name] = fields

        # Print a compact schema to console
        print("\n" + "=" * 80)
        print(full_name)
        for field in fields:
            print(f"- {field['name']} : {field['type']} (nullable={field['nullable']})")

        # Print 2 sample rows to console
        print("Sample rows:")
        df.show(2, truncate=False)

        # Save 2 sample rows to a per-table file
        # We store as JSON array for easy reading later
        sample_rows = df.limit(2).toJSON().collect()
        sample_rows = [json.loads(r) for r in sample_rows]

        safe_name = full_name.replace(".", "_")
        sample_path = sample_dir / f"{safe_name}.json"

        with open(sample_path, "w") as sf:
            json.dump(sample_rows, sf, indent=2, ensure_ascii=False)

        print(f"Sample saved to: {sample_path}")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Skip {full_name}, failed to read: {e}")

# Save all schemas to local json for later lookup
out_path = REPO_ROOT / "data" / "all_table_schemas.json"
with open(out_path, "w") as f:
    json.dump(all_schemas, f, indent=2, ensure_ascii=False)

print(f"\nSchemas saved to: {out_path}")
