import re


def snake2camel(snake_text: str):
    """
    >>> snake2camel("simple_test")
    'SimpleTest'
    """
    return "".join(word.title() for word in snake_text.split("_"))


CAP_LETTER_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")


def camel2snake(camel_text: str):
    """
    >>> camel2snake("SimpleCase")
    'simple_case'

    >>> camel2snake("simpleCase")
    'simple_case'
    """

    return CAP_LETTER_PATTERN.sub("_", camel_text).lower()
