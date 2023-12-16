import os
from pathlib import *
import shutil as sh

workspace_folder = Path().absolute()
build_dir = workspace_folder/'build'


def compile_and_run(source: Path, output: Path, compile_flags: list):
    flag_string = ' '.join(f'-{f}' for f in compile_flags)

    os.system(f'gcc {source} {flag_string} -o {source.stem}')
    os.system(f'./{source.parent/source.stem} >> {output}')


if __name__ == '__main__':
    # Инициализируем директорию сборки
    if build_dir.exists():
        sh.move(build_dir, workspace_folder/'backup')
        os.rmdir(build_dir)
    os.mkdir(build_dir)

    output_file = workspace_folder/'output.csv'








