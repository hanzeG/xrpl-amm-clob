import json
from pathlib import Path

import delta_sharing

# Path to Delta Sharing profile (repo-root relative)
REPO_ROOT = Path(__file__).resolve().parents[2]
profile_path = REPO_ROOT / "data" / "config.share"
client = delta_sharing.SharingClient(str(profile_path))

catalog = {}

shares = client.list_shares()
for share in shares:
    share_name = share.name
    catalog[share_name] = {}

    schemas = client.list_schemas(share)
    for schema in schemas:
        schema_name = schema.name
        catalog[share_name][schema_name] = []

        tables = client.list_tables(schema)
        for table in tables:
            catalog[share_name][schema_name].append(table.name)

print(json.dumps(catalog, indent=2, ensure_ascii=False))
