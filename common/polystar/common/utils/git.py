from os import popen


def get_git_username() -> str:
    return popen("git config user.name").read().strip()
