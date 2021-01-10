import logging

import usb.core
import usb.util

from polystar.common.communication.target_sender_abc import TargetSenderABC
from polystar.common.target_pipeline.target_abc import TargetABC, SimpleTarget

logger = logging.getLogger(__name__)


class USBConnectionFailed(Exception):
    pass


class USBTargetSender(TargetSenderABC):
    def __init__(self):
        self._endpoint = self._get_usb_endpoint()

    def send(self, target: TargetABC):
        data = target.to_json()
        if self._endpoint is not None:
            self._endpoint.write(target.to_json())
        else:
            logger.warning(f"{data} not sent")

    def _get_usb_endpoint(self) -> usb.core.Endpoint:
        try:
            dev = usb.core.find(idVendor=0xFFFE, idProduct=0x0001)

            if dev is None:
                raise USBConnectionFailed

            # With no arguments, the first configuration will be the active one
            dev.set_configuration()

            intf = dev.get_active_configuration()[(0, 0)]
            ep = usb.util.find_descriptor(
                intf,
                # match the first OUT endpoint
                custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT,
            )

            if ep is None:
                raise USBConnectionFailed()

            return ep
        except USBConnectionFailed:
            return self._handle_no_connection()

    @staticmethod
    def _handle_no_connection():
        logger.warning("failed to setup usb connection")
        # TODO: what should we do in production ?
        return None


if __name__ == "__main__":
    USBTargetSender().send(SimpleTarget(10, 20, 30))
