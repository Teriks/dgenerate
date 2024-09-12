import glob
import shutil
import os
import build.__main__

os.environ['DGENERATE_POETRY_LOCKFILE_PATH'] = os.path.join(os.path.dirname(__file__), 'poetry', 'poetry.lock')
os.environ['DGENERATE_POETRY_PYPROJECT_PATH'] = os.path.join(os.path.dirname(__file__), 'poetry', 'pyproject.toml')

platforms = {
    'Windows': ['win_amd64'],
    'Darwin': ['macosx_11_0_arm64'],
    'Linux': ['any']
}


def latest_file(directory):
    files = glob.glob(os.path.join(directory, '*'))

    if not files:
        return None

    return max(files, key=os.path.getmtime)


for platform, platform_tags in platforms.items():
    os.environ['DGENERATE_PLATFORM'] = platform

    for platform_tag in platform_tags:
        os.environ['DGENERATE_PLATFORM_TAG'] = platform_tag

        directory = os.path.join(os.path.dirname(__file__), 'dist', platform, platform_tag)
        os.makedirs(directory, exist_ok=True)
        build.__main__.main(['--wheel', '-o', directory])
        wheel = latest_file(directory)
        wheel_name = wheel.replace('any', platform_tag)
        dest = os.path.join(os.path.dirname(__file__), 'dist', os.path.basename(wheel_name))
        if os.path.exists(dest):
            os.unlink(dest)
        shutil.move(wheel, dest)
        shutil.rmtree(os.path.dirname(directory), ignore_errors=True)

build.__main__.main(['--sdist'])
