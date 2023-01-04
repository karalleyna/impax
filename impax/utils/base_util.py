from impax.utils.base.standard_file_system import StandardFileSystem
from impax.utils.base.simple_log import SimpleLog

ENVIRONMENT = "EXTERNAL"

if ENVIRONMENT == "GOOGLE":
    raise ValueError("Google file-system and logging no longer supported.")
elif ENVIRONMENT == "EXTERNAL":
    FS = StandardFileSystem()
    LOG = SimpleLog()
else:
    raise ValueError(f"Unrecognized library mode: {ENVIRONMENT}.")
