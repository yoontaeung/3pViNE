"""Ed25519 authentication for ZK-VNE control-plane HTTP messages."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


AUTH_DOMAIN = "zk-vne-http-signature-v1"
SIGNER_HEADER = "X-ZK-VNE-Signer"
MESSAGE_ID_HEADER = "X-ZK-VNE-Message-ID"
SIGNATURE_HEADER = "X-ZK-VNE-Signature"


class AuthenticationError(Exception):
    """Raised when a signed request cannot be authenticated."""


class AuthorizationError(Exception):
    """Raised when an authenticated identity cannot perform an operation."""


@dataclass(frozen=True)
class AuthenticatedIdentity:
    signer_id: str
    role: str
    message_id: str


def canonical_request(method: str, path: str, payload: Any) -> bytes:
    """Return the exact bytes signed by every protocol actor."""
    value = {
        "domain": AUTH_DOMAIN,
        "method": method.upper(),
        "path": path,
        "payload": payload,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class RequestSigner:
    def __init__(self, signer_id: str, private_key: Ed25519PrivateKey) -> None:
        self.signer_id = signer_id
        self.private_key = private_key

    @classmethod
    def from_pem(cls, signer_id: str, path: Path) -> "RequestSigner":
        private_key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError(f"{path} is not an Ed25519 private key")
        return cls(signer_id, private_key)

    def headers(self, method: str, path: str, payload: Any) -> dict[str, str]:
        message = canonical_request(method, path, payload)
        signature = self.private_key.sign(message)
        message_id = hashlib.sha256(
            self.signer_id.encode("utf-8") + b"\x00" + message
        ).hexdigest()
        return {
            SIGNER_HEADER: self.signer_id,
            MESSAGE_ID_HEADER: message_id,
            SIGNATURE_HEADER: base64.b64encode(signature).decode("ascii"),
        }


class RequestAuthenticator:
    def __init__(
        self,
        identities: dict[str, tuple[str, Ed25519PublicKey]],
    ) -> None:
        self.identities = identities

    @classmethod
    def from_config(cls, path: Path) -> "RequestAuthenticator":
        value = json.loads(path.read_text(encoding="utf-8"))
        identities: dict[str, tuple[str, Ed25519PublicKey]] = {}
        for signer_id, config in value["signers"].items():
            public_key_path = path.parent / config["public_key_file"]
            public_key = serialization.load_pem_public_key(public_key_path.read_bytes())
            if not isinstance(public_key, Ed25519PublicKey):
                raise ValueError(f"{public_key_path} is not an Ed25519 public key")
            identities[signer_id] = (str(config["role"]), public_key)
        return cls(identities)

    def verify(
        self,
        *,
        method: str,
        path: str,
        payload: Any,
        headers: Mapping[str, str],
    ) -> AuthenticatedIdentity:
        signer_id = headers.get(SIGNER_HEADER)
        message_id = headers.get(MESSAGE_ID_HEADER)
        encoded_signature = headers.get(SIGNATURE_HEADER)
        if not signer_id or not message_id or not encoded_signature:
            raise AuthenticationError("missing ZK-VNE authentication headers")
        try:
            role, public_key = self.identities[signer_id]
        except KeyError as exc:
            raise AuthenticationError(f"unknown signer {signer_id}") from exc

        message = canonical_request(method, path, payload)
        expected_message_id = hashlib.sha256(
            signer_id.encode("utf-8") + b"\x00" + message
        ).hexdigest()
        if message_id != expected_message_id:
            raise AuthenticationError("authentication message ID does not match request")
        try:
            signature = base64.b64decode(encoded_signature, validate=True)
            public_key.verify(signature, message)
        except (binascii.Error, ValueError, InvalidSignature) as exc:
            raise AuthenticationError("invalid request signature") from exc
        return AuthenticatedIdentity(
            signer_id=signer_id,
            role=role,
            message_id=message_id,
        )


def require_role(identity: AuthenticatedIdentity, role: str) -> None:
    if identity.role != role:
        raise AuthorizationError(
            f"signer {identity.signer_id} has role {identity.role}, not {role}"
        )


def signer_from_environment() -> RequestSigner:
    return RequestSigner.from_pem(
        os.environ["ZK_VNE_SIGNER_ID"],
        Path(os.environ["ZK_VNE_PRIVATE_KEY"]),
    )


def authenticator_from_environment() -> RequestAuthenticator:
    return RequestAuthenticator.from_config(Path(os.environ["ZK_VNE_TRUST_CONFIG"]))
