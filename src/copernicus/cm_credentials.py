from pathlib import Path


class CMCredentials:
    """
    Check the presence of Copernicus Marine credentials at a single path.
    """

    def __init__(
        self, path: str | Path = "~/.copernicusmarine/.copernicusmarine-credentials"
    ):
        self.path = Path(path).expanduser()

    def ensure_present(self) -> Path:
        """Raise if no credentials found; return the resolved path otherwise."""
        if not self.path.is_file():
            raise RuntimeError(
                f"Copernicus Marine credentials not found at: {self.path}\n"
                "Please perform a one-off 'copernicusmarine login' or "
                "provide the correct credentials path."
            )
        return self.path
