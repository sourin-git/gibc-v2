import pytest

from gibc.source_filter import is_wikipedia_source, normalized_hostname


@pytest.mark.parametrize(
    "url",
    [
        "https://wikipedia.org/",
        "https://en.wikipedia.org/wiki/Physics",
        "http://en.m.wikipedia.org/wiki/Physics",
        "https://EN.WikiPedia.ORG/wiki/Physics",
        "https://en.wikipedia.org./wiki/Physics",
        "https://en.wikipedia.org:443/wiki/Physics",
        "https://user:pass@de.wikipedia.org/wiki/Physik",
        "  https://fr.wikipedia.org/wiki/Physique  ",
    ],
)
def test_wikipedia_hosts_excluded(url):
    assert is_wikipedia_source(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://notwikipedia.org/page",
        "https://wikipedia.org.example.com/page",
        "https://example.com/wikipedia.org/page",
        "https://example.com/?ref=en.wikipedia.org",
        "https://en.wikibooks.org/wiki/Physics",
        "https://commons.wikimedia.org/wiki/File:X.png",
        "https://wikipedia.com/",
        "",
        None,
        "not a url",
        "http://[::1",
    ],
)
def test_other_urls_kept(url):
    assert not is_wikipedia_source(url)


def test_normalized_hostname():
    assert normalized_hostname("HTTPS://User@En.Wikipedia.Org.:8080/x") == "en.wikipedia.org"
    assert normalized_hostname("no-scheme.example.com/path") is None
