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
import re
import secrets
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

# The namespace used when a deployment has not defined any.
DEFAULT_NAMESPACE = "default"
# A namespace value meaning "every namespace this principal could ever see".
ALL_NAMESPACES = "*"

# A namespace is an identifier, not free text. Constraining it is not cosmetic:
# the namespace reaches a metric label, a cache partition and a retrieval
# filter, and an unconstrained one lets a caller mint a new series on every
# request until the process runs out of memory. Sixty three characters is the
# usual label length ceiling and is far more than a tenant name needs.
NAMESPACE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
MAX_NAMESPACE_LENGTH = 63


class InvalidNamespace(ValueError):
    """A namespace that is not a well formed identifier."""


def normalise_namespace(value: str | None) -> str:
    """Return ``value`` as a canonical namespace, or raise.

    Case is folded because two tenants differing only in case are two storage
    partitions that read as one to a human, which is the kind of ambiguity a
    tenancy boundary must not have. The wildcard is rejected here because it is
    a grant a principal holds, never a namespace documents live in: accepting it
    as a target would silently address a partition that can never match.
    """
    # Stripped before the emptiness test so a blank field and an absent one are
    # the same request. A form that posts "   " means "I did not choose".
    candidate = (value or "").strip().lower() or DEFAULT_NAMESPACE
    if candidate == ALL_NAMESPACES:
        raise InvalidNamespace("the wildcard is a grant, not a namespace")
    if not NAMESPACE_PATTERN.fullmatch(candidate):
        raise InvalidNamespace(
            "a namespace must be 1 to "
            f"{MAX_NAMESPACE_LENGTH} characters of a-z, 0-9, hyphen or underscore, "
            "starting with a letter or digit"
        )
    return candidate


def normalise_grant(value: str) -> str:
    """Return ``value`` as a namespace a principal may be granted.

    Unlike a request target this accepts the wildcard, because granting "every
    namespace" is exactly what an administrator credential needs to express.
    """
    candidate = value.strip().lower()
    if candidate == ALL_NAMESPACES:
        return ALL_NAMESPACES
    return normalise_namespace(candidate)


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
    # Set by the store from the credential itself. The name is a label an
    # operator chooses and may reuse; anything keyed on it (a rate limit bucket,
    # a per caller budget) would silently pool two distinct credentials into one
    # allowance. This is derived from the key hash, so it is unique per key.
    key_id: str = ""
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

    def caller_id(self) -> str:
        """The key for anything that meters this caller.

        Falls back to the name only in open mode, where there is one implicit
        local principal and therefore nothing to confuse it with.
        """
        return self.key_id or self.name

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

    def __init__(
        self,
        principals: dict[str, Principal] | None = None,
        misconfigured: str = "",
    ) -> None:
        self._by_hash: dict[str, Principal] = dict(principals or {})
        # Non empty when a principal table was supplied and could not be used.
        # That is a different state from "none supplied" and must not resolve to
        # the same open access, or a typo in the table silently unlocks the
        # deployment that was configuring authentication.
        self.misconfigured = misconfigured

    @property
    def enabled(self) -> bool:
        return bool(self._by_hash)

    @property
    def count(self) -> int:
        return len(self._by_hash)

    def add(self, raw_key: str, principal: Principal) -> None:
        digest = hash_key(raw_key)
        # Truncated so the identity can appear in a log line or a bucket table
        # without carrying the full stored hash around. Sixty four bits is far
        # beyond collision range for a hand written principal table.
        principal.key_id = digest[:16]
        self._by_hash[digest] = principal

    def resolve(self, raw_key: str | None) -> Principal:
        """Return the principal for a key, or raise.

        When the store is empty the caller is an administrator over every
        namespace, which is what makes the single machine path work with no
        configuration at all.
        """
        if self.misconfigured:
            # Fail closed. An operator who configured a table and got it wrong
            # wanted authentication, so the safe reading of a broken table is
            # "nobody gets in" rather than "everybody does".
            raise UnauthorizedError("the principal table could not be loaded")
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
            return cls(misconfigured=f"the principal table is not valid JSON: {exc}")

        store = cls()
        for raw_key, spec in (payload or {}).items():
            if not isinstance(spec, dict):
                logger.warning("skipping principal %r: not an object", raw_key)
                continue
            try:
                principal = Principal(
                    name=str(spec.get("name", "unnamed")),
                    role=Role(str(spec.get("role", "reader"))),
                    namespaces=[
                        normalise_grant(str(entry))
                        for entry in (spec.get("namespaces") or [DEFAULT_NAMESPACE])
                    ],
                    requests_per_minute=int(spec.get("requests_per_minute", 0)),
                    daily_token_budget=int(spec.get("daily_token_budget", 0)),
                )
            except (ValueError, TypeError) as exc:
                logger.warning("skipping principal %r: %s", spec.get("name", raw_key), exc)
                continue
            store.add(str(raw_key), principal)

        if not store.enabled:
            # Something was supplied but nothing usable came out of it, so every
            # entry was skipped. Same reasoning as unparseable JSON.
            store.misconfigured = "the principal table contained no usable entries"
        return store

    @classmethod
    def from_file(cls, path: Path) -> PrincipalStore:
        """Load a principal table from a file that was explicitly configured.

        Naming a file is itself the intent to authenticate, so a missing or
        unreadable one fails closed. A secret that did not mount is the exact
        case this protects: the file is absent, and treating absence as "no
        authentication wanted" would open the deployment at the worst moment.
        """
        if not path.exists():
            logger.error("the configured principal file does not exist: %s", path)
            return cls(misconfigured=f"the principal file {path} does not exist")
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.error("could not read the principal file %s: %s", path, exc)
            return cls(misconfigured=f"the principal file {path} could not be read")

    def describe(self) -> dict[str, object]:
        if self.misconfigured:
            return {
                "authentication": "failed closed (misconfigured)",
                "detail": self.misconfigured,
                "principals": 0,
                "roles": [],
            }
        return {
            "authentication": "enabled" if self.enabled else "disabled (open access)",
            "principals": self.count,
            "roles": sorted({p.role.value for p in self._by_hash.values()}),
        }
