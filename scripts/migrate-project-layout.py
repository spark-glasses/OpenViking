"""Run with OV/backend writers stopped and a private backup; default is dry-run."""
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
parser.add_argument("--journal", required=True)
parser.add_argument("--apply", action="store_true")
parser.add_argument("--questions", action="store_true", help="Upgrade question records after migrating Project folders")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["OPENVIKING_CONFIG_FILE"] = str(Path(args.config).resolve())

from openviking.service.core import OpenVikingService
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier
from openviking.session.memory.project_layout_migration import plan_migration, apply_migration
if args.questions:
    from openviking.session.memory.question_layout_migration import plan_migration

async def main():
    user = UserIdentifier(args.account, args.user)
    service = OpenVikingService(user=user)
    await service.initialize(start_background_workers=False)
    try:
        ctx = RequestContext(user=user, role=Role.ROOT)
        path = Path(args.journal)
        plan = json.loads(path.read_text()) if path.exists() else await plan_migration(service.viking_fs, ctx)
        print(json.dumps({"status": plan["status"], "moves": len(plan["moves"]), "files": len(plan["files"])}))
        if args.apply:
            await apply_migration(service.viking_fs, ctx, plan, path, service.vikingdb_manager)
            print(json.dumps({"status": plan["status"]}))
    finally:
        await service.close()

asyncio.run(main())
