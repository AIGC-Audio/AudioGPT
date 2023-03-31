import os
import subprocess


def link_file(from_file, to_file):
    subprocess.check_call(
        f'ln -s "`realpath --relative-to="{os.path.dirname(to_file)}" "{from_file}"`" "{to_file}"', shell=True)


def move_file(from_file, to_file):
    subprocess.check_call(f'mv "{from_file}" "{to_file}"', shell=True)


def copy_file(from_file, to_file):
    subprocess.check_call(f'cp -r "{from_file}" "{to_file}"', shell=True)


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)