import multiprocessing
import time
from multiprocessing import Process
from pathlib import Path
from subprocess import PIPE, run

data_path = Path("./impax/data")

msh2msh = "./impax/gaps/bin/x86_64/msh2msh"
msh2df = "./impax/gaps/bin/x86_64/msh2df"
grd2msh = "./impax/gaps/bin/x86_64/grd2msh"

files = list(data_path.glob("*/*/*.obj"))


def process(s, e):
    tmp_g = f"./tmp.grd{s}"
    for i in range(s, e):
        file = files[i]
        s = time.time()
        
        result1 = run(
            [msh2df, str(file), tmp_g, "-estimate_sign", "-spacing", "0.005", "-v"],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

        result2 = run(
            [grd2msh, tmp_g, str(file)[:-3] + "ply"],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

        result3 = run(["rm", tmp_g])

        if result1.returncode == 0 and result2.returncode == 0 and result3.returncode == 0:
            print(f"{i} - Done in {time.time() - s} - {str(file)}")
            result3 = run(["rm", str(file)])


if __name__ == "__main__":
    n_process = multiprocessing.cpu_count()
    n_file = len(files)
    n_per = n_file // n_process

    ps = []
    for i in range(n_process):
        p = Process(target=process, args=(i * n_per, min((i + 1) * n_per, n_file)))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
