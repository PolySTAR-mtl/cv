import sys

from research.dataset.twitch.make_thumbnails import ThumbnailsGenerator

if __name__ == '__main__':
    _video_name = sys.argv[1]
    print(f'Fragmenting video {_video_name}')
    ThumbnailsGenerator(_video_name).run()
