import sys

from research.common.dataset.twitch.robots_views_extractor import RobotsViewExtractor

if __name__ == "__main__":
    for _video_name in sys.argv[1:]:
        RobotsViewExtractor(_video_name).run()
