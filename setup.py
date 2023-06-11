from setuptools import setup
import subprocess
import os

deepsensor_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

assert "-" not in deepsensor_version
assert "." in deepsensor_version

assert os.path.isfile("deepsensor/version.py")
with open("deepsensor/VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % deepsensor_version)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        package_data={"deepsensor": ["VERSION"]},
        include_package_data=True,
    )
