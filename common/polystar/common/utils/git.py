from os import popen


def get_git_username() -> str:
    return popen("git config user.name").read().strip()


if __name__ == "__main__":
    print(f"'{get_git_username()}'")
