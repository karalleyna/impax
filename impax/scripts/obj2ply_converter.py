from pathlib import Path
from subprocess import PIPE, run

data_path = Path("/mnt/e/Desktop/repos/impax/impax/data")

msh2msh = "/mnt/e/Desktop/repos/impax/impax/gaps/bin/x86_64/msh2msh"

for file in data_path.glob("*/*/*.obj"):
    result = run([msh2msh, str(file), str(file)[:-3] + "ply"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
