"""Module for loading and accessing MITRE ATT&CK data."""

import os
from enum import StrEnum

from mitreattack.stix20 import MitreAttackData
from pydantic import BaseModel, Field

stix_filepath = os.environ.get("STIX_BUNDLE", "./resources/enterprise-attack.json")
mitre_attack_data = MitreAttackData(stix_filepath=stix_filepath)

tactics_objs = mitre_attack_data.get_tactics()
tactics = [tactic["name"] for tactic in tactics_objs]

techniques = mitre_attack_data.get_techniques()
techniques = [technique["name"] for technique in techniques]

Tactic = StrEnum("Tactic", {tactic.replace(" ", "_").upper(): tactic for tactic in tactics})
Technique = StrEnum("Technique", {technique.replace(" ", "_").upper(): technique for technique in techniques})


class SessionTTPs(BaseModel):
    """List of MITRE ATT&CK tactics and techniques observed in a log session.

    Each session contains multiple log events, to which multiple tactics and techniques may apply.
    Only tactics and techniques that are confidently observed in the session should be included.
    """

    tactics: list[Tactic] = Field(  # pyright: ignore[reportInvalidTypeForm]
        description="List of MITRE ATT&CK tactics observed in the session.",
        examples=[t.value for t in Tactic][0:3],  # pyright: ignore[reportAttributeAccessIssue]
    )

    techniques: list[Technique] = Field(  # pyright: ignore[reportInvalidTypeForm]
        description="List of MITRE ATT&CK techniques observed in the session.",
        examples=[t.value for t in Technique][0:3],  # pyright: ignore[reportAttributeAccessIssue]
    )
