"""Share one KVMem host budget (H) across N concurrent sessions.

KVMem's own model is: B/R are per request, H is a SHARED budget. The llama worker hosts one
process per model, so it can hand the whole H to its single conversation store; vLLM serves C
requests from one EngineCore, so the H budget has to be divided explicitly and enforced, or the
process RSS grows past the configured limit without anyone noticing.

Allocation policy: every live session gets `H // sessions` (the retained-history slot count), and
a session that needs more may borrow only from the *unused* remainder. Over-subscription fails
loudly instead of silently over-committing.
"""
from dataclasses import dataclass


class BudgetError(RuntimeError):
    pass


@dataclass
class SessionGrant:
    session: int
    capacity_bytes: int
    used_bytes: int = 0

    @property
    def free_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.used_bytes)


class SessionBudgetPool:
    """Divides `H` (bytes) across the live sessions of one EngineCore."""

    def __init__(self, host_bytes, sessions):
        if host_bytes <= 0:
            raise ValueError("the host budget must be positive")
        if sessions <= 0:
            raise ValueError("sessions must be positive")
        self.host_bytes = int(host_bytes)
        self.sessions = int(sessions)
        self._grants: dict[int, SessionGrant] = {}

    # --- membership -------------------------------------------------------------------
    def open(self, session):
        """Give a session its fair share (H // sessions)."""
        session = int(session)
        if session in self._grants:
            return self._grants[session]
        share = self.host_bytes // self.sessions
        self._grants[session] = SessionGrant(session, share)
        return self._grants[session]

    def close(self, session):
        """Release a session; borrowed remainder is returned to the pool automatically."""
        self._grants.pop(int(session), None)

    def grant(self, session):
        return self._grants[int(session)]

    # --- accounting -------------------------------------------------------------------
    def charge(self, session, nbytes):
        """Charge a session; it may borrow only the unassigned remainder."""
        grant = self.grant(session)
        nbytes = int(nbytes)
        if nbytes <= 0:
            return grant
        if grant.used_bytes + nbytes <= grant.capacity_bytes:
            grant.used_bytes += nbytes
            return grant
        unassigned = self.unassigned_bytes()
        borrow = (grant.used_bytes + nbytes) - grant.capacity_bytes
        if borrow > unassigned:
            raise BudgetError(
                f"session {session} needs {nbytes} B, {grant.free_bytes} B free in its share and "
                f"only {unassigned} B unassigned ({self.used_bytes()}/{self.host_bytes} B in use)")
        grant.capacity_bytes += borrow
        grant.used_bytes += nbytes
        return grant

    def release(self, session, nbytes):
        grant = self.grant(session)
        grant.used_bytes = max(0, grant.used_bytes - int(nbytes))
        return grant

    def assigned_bytes(self):
        return sum(g.capacity_bytes for g in self._grants.values())

    def used_bytes(self):
        return sum(g.used_bytes for g in self._grants.values())

    def unassigned_bytes(self):
        return max(0, self.host_bytes - self.assigned_bytes())

    def fits(self, nbytes):
        return self.used_bytes() + int(nbytes) <= self.host_bytes
