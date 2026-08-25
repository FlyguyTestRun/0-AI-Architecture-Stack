"""Security layer: identity, roles, tenancy and rate limiting.

Everything here is opt in. A small team running this on one machine should not
have to configure authentication to ask a question, and an enterprise deployment
should not have to bolt tenancy on afterwards. The same code serves both: with
no principals configured the API is open exactly as before, and configuring one
turns every boundary on at once.
"""

from zerostack.security.identity import (
    ForbiddenError,
    Principal,
    PrincipalStore,
    Role,
    UnauthorizedError,
)
from zerostack.security.limits import RateLimiter, RateLimitExceeded

__all__ = [
    "ForbiddenError",
    "Principal",
    "PrincipalStore",
    "RateLimitExceeded",
    "RateLimiter",
    "Role",
    "UnauthorizedError",
]
