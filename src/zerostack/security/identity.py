"""API keys, roles and namespace scoping.

The tenancy primitive is the namespace, matching the model the enterprise tier
of this architecture already uses, so a corpus organised here transfers upward
without being reorganised.

A principal is an API key with a role and a set of namespaces. Roles decide what
kind of operation is allowed; namespaces decide which documents it may touch.
Keeping those separate matters: a developer who may ingest documents still must
not read another department's corpus, and collapsing the two into one concept
makes that combination inexpressible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

# The namespace used when a deployment has not defined any.
DEFAULT_NAMESPACE = "default"
# A namespace value meaning "every namespace this principal could ever see".
ALL_NAMESPACES = "*"


class UnauthorizedError(RuntimeError):
    """No valid credential was presented."""


class ForbiddenError(RuntimeError):
    """A valid credential that is not permitted this operation or namespace."""


class Role(StrEnum):
    """What a principal may do.

    Ordered from least to most capable. Each role includes the permissions of
    those before it, which keeps the check a comparison rather than a matrix
    nobody can reason about.
    """

    READER = "reader"
    WRITER = "writer"
    OPERATOR = "operator"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return _ROLE_ORDER.index(self)

    def allows(self, required: Role) -> bool:
        return self.rank >= required.rank


_ROLE_ORDER = [Role.READER, Role.WRITER, Role.OPERATOR, Role.ADMIN]


@dataclass
class Principal:
    """An authenticated caller."""

    name: str
    role: Role = Role.READER
    namespaces: list[str] = field(default_factory=lambda: [DEFAULT_NAMESPACE])
    # Per principal ceilings. Zero means "use the deployment default".
    requests_per_minute: int = 0
    daily_token_budget: int = 0

    def may(self, required: Role) -> bool:
        return self.role.allows(required)

    def sees_all_namespaces(self) -> bool:
        return ALL_NAMESPACES in self.namespaces

    def may_access(self, namespace: str) -> bool:
        return self.sees_all_namespaces() or namespace in self.namespaces

    def default_namespace(self) -> str:
        """The namespace to use when a request does not name one."""
        if self.sees_all_namespaces():
            return DEFAULT_NAMESPACE
        return self.namespaces[0] if self.namespaces else DEFAULT_NAMESPACE

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "namespaces": list(self.namespaces),
        }


def hash_key(raw_key: str) -> str:
    """Hash an API key for storage.

    Keys are stored hashed so that a leaked configuration file does not hand over
    working credentials. SHA-256 without a per key salt is deliberate here: the
    input is a high entropy random token rather than a human chosen password, so
    the salt would defend against nothing while making rotation harder.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_key(prefix: str = "zs") -> str:
    """Generate a key with enough entropy that guessing is not a threat."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


class PrincipalStore:
    """Resolves an API key to a principal.

    An empty store means authentication is disabled and every caller is treated
    as a local administrator. That is the right default for a single machine
    deployment and the wrong one for anything exposed, which is why the health
    output states plainly which mode is active rather than leaving it implied.
    """

    def __init__(self, principals: dict[str, Principal] | None = None) -> None:
        self._by_hash: dict[str, Principal] = dict(principals or {})

    @property
    def enabled(self) -> bool:
        return bool(self._by_hash)

    @property
    def count(self) -> int:
        return len(self._by_hash)

    def add(self, raw_key: str, principal: Principal) -> None:
        self._by_hash[hash_key(raw_key)] = principal

    def resolve(self, raw_key: str | None) -> Principal:
        """Return the principal for a key, or raise.

        When the store is empty the caller is an administrator over every
        namespace, which is what makes the single machine path work with no
        configuration at all.
        """
        if not self.enabled:
            return Principal(
                name="local",
                role=Role.ADMIN,
                namespaces=[ALL_NAMESPACES],
            )
        if not raw_key:
            raise UnauthorizedError("an API key is required")

        presented = hash_key(raw_key)
        for stored_hash, principal in self._by_hash.items():
            # Compared in constant time so that response timing does not leak
            # how much of a guessed key was correct.
            if hmac.compare_digest(stored_hash, presented):
                return principal
        raise UnauthorizedError("unrecognised API key")

    @classmethod
    def from_json(cls, raw: str) -> PrincipalStore:
        """Build a store from a JSON mapping of key to principal.

        Shape: {"zs_abc...": {"name": "hr-app", "role": "writer",
                              "namespaces": ["hr"], "requests_per_minute": 60}}

        A malformed entry is skipped with a warning rather than raising, so one
        bad entry cannot stop the service from starting and lock everyone out.
        """
        if not raw.strip():
            return cls()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("could not parse the principal table: %s", exc)
            return cls()

        store = cls()
        for raw_key, spec in (payload or {}).items():
            if not isinstance(spec, dict):
                logger.warning("skipping principal %r: not an object", raw_key)
                continue
            try:
                principal = Principal(
                    name=str(spec.get("name", "unnamed")),
                    role=Role(str(spec.get("role", "reader"))),
                    namespaces=list(spec.get("namespaces") or [DEFAULT_NAMESPACE]),
                    requests_per_minute=int(spec.get("requests_per_minute", 0)),
                    daily_token_budget=int(spec.get("daily_token_budget", 0)),
                )
            except (ValueError, TypeError) as exc:
                logger.warning("skipping principal %r: %s", spec.get("name", raw_key), exc)
                continue
            store.add(str(raw_key), principal)
        return store

    @classmethod
    def from_file(cls, path: Path) -> PrincipalStore:
        if not path.exists():
            return cls()
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.error("could not read the principal file %s: %s", path, exc)
            return cls()

    def describe(self) -> dict[str, object]:
        return {
            "authentication": "enabled" if self.enabled else "disabled (open access)",
            "principals": self.count,
            "roles": sorted({p.role.value for p in self._by_hash.values()}),
        }
