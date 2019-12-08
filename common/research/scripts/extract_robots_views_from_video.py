import sys

from research.dataset.twitch.robots_views_extractor import RobotsViewExtractor

if __name__ == '__main__':
    _video_name = sys.argv[1]
    RobotsViewExtractor(_video_name).run()
