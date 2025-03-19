import contextlib
import glob
import shutil
import os
import subprocess
from importlib.machinery import SourceFileLoader
import sys
import dgenerate.resources as _resources

project_dir = os.path.dirname(os.path.abspath(__file__))

setup = SourceFileLoader(
    'setup_as_library', os.path.join(project_dir, 'setup.py')).load_module()


def latest_file(directory):
    files = glob.glob(os.path.join(directory, '*'))

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def commit_and_branch() -> tuple[str, str]:
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()
    return commit, branch


@contextlib.contextmanager
def with_release_data(directory, pre_release: bool, git_commit_and_branch: tuple[str, str] = None):
    dgenerate_release_data = os.path.join(directory, 'release.json')

    if os.path.exists(dgenerate_release_data):
        os.unlink(dgenerate_release_data)

    if git_commit_and_branch is not None:
        commit, branch = git_commit_and_branch
    else:
        commit, branch = commit_and_branch()
    version = setup.VERSION

    info = _resources.CurrentReleaseInfo(
        version=version,
        commit=commit,
        branch=branch,
        pre_release=pre_release
    )

    with open(dgenerate_release_data, 'w') as switch_file:
        info.json_dump(switch_file)

    yield

    if os.path.exists(dgenerate_release_data):
        os.unlink(dgenerate_release_data)


@contextlib.contextmanager
def with_dir_clean(dirs: list):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    yield
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def main():
    global project_dir

    os.environ['DGENERATE_POETRY_LOCKFILE_PATH'] = os.path.join(project_dir, 'poetry', 'poetry.lock')
    os.environ['DGENERATE_POETRY_PYPROJECT_PATH'] = os.path.join(project_dir, 'poetry', 'pyproject.toml')

    platforms = {
        'Windows': ['win_amd64'],
        'Darwin': ['macosx_11_0_arm64'],
        'Linux': ['any']
    }

    os.chdir(project_dir)

    import build.__main__

    dist_dir = os.path.join(project_dir, 'dist')
    build_dir = os.path.join(project_dir, 'build')
    egg_info = os.path.join(project_dir, 'dgenerate.egg-info')
    os.makedirs(dist_dir, exist_ok=True)

    c_and_b = commit_and_branch()

    commit, branch = c_and_b

    pre_release = '--pre-release' in sys.argv

    with with_release_data('dgenerate', pre_release=pre_release, git_commit_and_branch=c_and_b), \
            with_dir_clean([build_dir, egg_info]):

        for platform, platform_tags in platforms.items():
            os.environ['DGENERATE_PLATFORM'] = platform

            for platform_tag in platform_tags:
                os.environ['DGENERATE_PLATFORM_TAG'] = platform_tag

                wheel_directory = os.path.join(dist_dir, platform, platform_tag)
                os.makedirs(wheel_directory, exist_ok=True)
                build.__main__.main(['--wheel', '-o', wheel_directory])
                wheel = latest_file(wheel_directory)
                wheel_name = wheel.replace('any', platform_tag)
                if pre_release:
                    wheel_name = wheel_name.replace('py3', f'{commit}-py3')
                dest = os.path.join('dist', os.path.basename(wheel_name))
                if os.path.exists(dest):
                    os.unlink(dest)
                print(f'Moving {wheel} -> {dest}')
                shutil.move(wheel, dest)
                shutil.rmtree(os.path.dirname(wheel_directory), ignore_errors=True)

        build.__main__.main(['--sdist'])


if __name__ != 'build_dist_as_library':
    main()
