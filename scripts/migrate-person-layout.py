"""Offline, explicitly scoped migration. Stop OV/backend writers and back up first."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--account", required=True)
parser.add_argument("--user", required=True)
parser.add_argument("--journal", required=True, help="Private durable JSON backup/receipt path")
parser.add_argument("--references", help="Reference map prepared by the backend DB migration")
parser.add_argument("--apply", action="store_true", help="Otherwise only report planned changes")
args = parser.parse_args()
# Execute this checkout, never a separately installed upstream wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["OPENVIKING_CONFIG_FILE"] = str(Path(args.config).resolve())

from openviking.service.core import OpenVikingService
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier
from openviking.session.memory.person_layout_migration import plan_migration, apply_migration

async def main():
    user = UserIdentifier(args.account, args.user)
    service = OpenVikingService(user=user)
    await service.initialize(start_background_workers=False)
    try:
        ctx = RequestContext(user=user, role=Role.ROOT)
        journal = Path(args.journal)
        plan = json.loads(journal.read_text()) if journal.exists() else await plan_migration(service.viking_fs, ctx, json.loads(Path(args.references).read_text()) if args.references else None)
        print(json.dumps({"mode": "apply" if args.apply else "dry-run", "user": args.user,
                          "moves": len(plan["moves"]), "changedFiles": len(plan["files"]), "status": plan["status"]}), flush=True)
        if args.apply:
            await apply_migration(service.viking_fs, ctx, plan, journal, service.vikingdb_manager)
            print(json.dumps({"status": plan["status"], "journal": str(journal)}), flush=True)
    finally:
        await service.close()

asyncio.run(main())
