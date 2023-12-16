import os
from pathlib import *
import shutil as sh

workspace_folder = Path().absolute()
build_dir = workspace_folder/'build'


def compile_and_run(source: Path, output: Path, compile_flags: list):
    flag_string = ' '.join(f'-{f}' for f in compile_flags if f is not None)

    compile_string = f'gcc {source} {flag_string} -o {build_dir/source.stem}'
    print('+ ' + compile_string)
    os.system(compile_string)
    run_string = f'{build_dir/source.stem} >> {output}'
    print('+ ' + run_string)
    os.system(run_string)


if __name__ == '__main__':
    # Инициализируем директорию сборки
    os.mkdir(build_dir)


    num_threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100, 120, 140, 160]
    dataset_sizes = ['MINI_DATASET', 'SMALL_DATASET', 'MEDIUM_DATASET', 'LARGE_DATASET']
    optimizations = [None, 'O2', 'O3', 'fast']

    source = workspace_folder/'for.c'

    for optimization in optimizations:
        output_file = build_dir/f'output_{optimization}.csv'
        for dataset in dataset_sizes:
            os.system(f"echo -n '{dataset},' >> {output_file}")
            for threads in num_threads:
                compile_and_run(source, output_file, ['fopenmp', optimization, f'DNUM_THREADS={threads}', f'D{dataset}'])
                os.system(f"echo -n  ',' >> {output_file}")
            os.system(f"echo '' >> {output_file}")








