from logging import Logger

from impax.utils.base.standard_file_system import StandardFileSystem

ENVIRONMENT = "EXTERNAL"

if ENVIRONMENT == "GOOGLE":
    raise ValueError("Google file-system and logging no longer supported.")
elif ENVIRONMENT == "EXTERNAL":
    FS = StandardFileSystem()
    LOG = Logger("simple logger")
else:
    raise ValueError(f"Unrecognized library mode: {ENVIRONMENT}.")
