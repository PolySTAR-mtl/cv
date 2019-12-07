from time import sleep

from research.constants import TWITCH_DSET
from research.dataset.twitch.robot_view import process_all_images_in_dir

if __name__ == '__main__':
    while 1:
        process_all_images_in_dir(TWITCH_DSET / 'raw-frames')
        sleep(.1)
