from pathlib import Path
from subprocess import PIPE, run

data_path = Path("/Users/burak/Desktop/repos/impax/impax/data")

msh2msh = "/Users/burak/Desktop/repos/impax/impax/gaps/bin/arm64/msh2msh"

for file in data_path.glob("*/*/*.obj"):
    result = run([msh2msh, str(file), str(file)[:-3] + "ply"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
