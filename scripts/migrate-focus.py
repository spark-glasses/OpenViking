"""Offline reviewed Project -> Focus/entity migration. Default is dry-run.

Stop OV/backend writers first. --decisions is a JSON object mapping each old
Project UUID to {"kind":"entity"} or {"kind":"focus","userIntent":"..."}.
The journal is a private rollback source; retain it after migration.
"""

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
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--decisions", help="Reviewed Project classifications")
mode.add_argument(
    "--lifecycle",
    action="store_true",
    help="Merge legacy Focus visibility/status into forming/active/archived",
)
parser.add_argument("--journal", required=True)
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["OPENVIKING_CONFIG_FILE"] = str(Path(args.config).resolve())

from openviking.service.core import OpenVikingService
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier
from openviking.session.memory.focus_migration import (
    plan_migration,
    plan_lifecycle_migration,
    apply_migration,
)


async def main():
    user = UserIdentifier(args.account, args.user)
    service = OpenVikingService(user=user)
    await service.initialize(start_background_workers=False)
    try:
        ctx = RequestContext(user=user, role=Role.ROOT)
        journal = Path(args.journal)
        plan = (
            json.loads(journal.read_text())
            if journal.exists()
            else await plan_lifecycle_migration(service.viking_fs, ctx)
            if args.lifecycle
            else await plan_migration(
                service.viking_fs, ctx, json.loads(Path(args.decisions).read_text())
            )
        )
        print(
            json.dumps(
                {"status": plan["status"], "moves": plan["moves"], "files": len(plan["files"])}
            )
        )
        if args.apply:
            await apply_migration(service.viking_fs, ctx, plan, journal, service.vikingdb_manager)
            print(json.dumps({"status": plan["status"]}))
    finally:
        await service.close()


asyncio.run(main())
