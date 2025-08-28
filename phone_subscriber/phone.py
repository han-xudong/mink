#!/usr/bin/env python

import re
import zmq
import pathlib
import numpy as np
from typing import Tuple
from datetime import datetime
from . import phone_msg_pb2


class PhonePublisher:
    """
    PhonePublisher class.

    This class is used to publish phone messages using ZeroMQ.

    Attributes:
        context (zmq.Context): The ZeroMQ context.
        publisher (zmq.Socket): The ZeroMQ publisher socket.
    """

    def __init__(
        self,
        host: str,
        port: str,
        hwm: int = 1,
        conflate: bool = True,
    ) -> None:
        """
        Publisher initialization.

        Args:
            host (str): The host address of the publisher.
            port (int): The port number of the publisher.
            hwm (int): High water mark for the publisher. Default is 1.
            conflate (bool): Whether to conflate messages. Default is True.
        """

        print("{:-^80}".format(" Phone Publisher Initialization "))
        print(f"Address: tcp://{host}:{port}")

        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ publisher
        self.publisher = self.context.socket(zmq.PUB)
        # Set high water mark
        self.publisher.set_hwm(hwm)
        # Set conflate
        self.publisher.setsockopt(zmq.CONFLATE, conflate)
        # Bind the address
        self.publisher.bind(f"tcp://{host}:{port}")

        # Read the protobuf definition for Phone message
        with open(
            pathlib.Path(__file__).parent / "protobuf/phone_msg.proto",
        ) as f:
            lines = f.read()
        messages = re.search(r"message\s+Phone\s*{{(.*?)}}", lines, re.DOTALL)
        body = messages.group(1)
        print("Message Phone")
        print("{\n" + body + "\n}")

        print("Phone Publisher Initialization Done.")
        print("{:-^80}".format(""))

    def publishMessage(
        self,
        color_img_bytes: bytes = b"",
        depth_img_bytes: bytes = b"",
        depth_width: int = 256,
        depth_height: int = 192,
        local_pose: list = np.zeros(6, dtype=np.float32).tolist(),
        global_pose: list = np.zeros(6, dtype=np.float32).tolist(),
    ) -> None:
        """
        Publish the message.

        Args:
            color_img_bytes: The image captured by the camera.
            depth_img: The depth image captured by the camera.
            pose: The pose of the marker (numpy array or list).
        """

        # Set the message
        phone = phone_msg_pb2.Phone()
        phone.timestamp = datetime.now().timestamp()
        phone.color_img = color_img_bytes
        phone.depth_img = depth_img_bytes
        phone.depth_width = depth_width
        phone.depth_height = depth_height
        phone.local_pose[:] = local_pose
        phone.global_pose[:] = global_pose

        # Publish the message
        self.publisher.send(phone.SerializeToString())

    def close(self):
        """
        Close ZMQ socket and context.
        """

        if hasattr(self, "publisher") and self.publisher:
            self.publisher.close()
        if hasattr(self, "context") and self.context:
            self.context.term()


class PhoneSubscriber:
    """
    PhoneSubscriber class.
    
    This class subscribes to messages from a publisher and parses the received messages.
    
    Attributes:
        context (zmq.Context): The ZeroMQ context for the subscriber.
        subscriber (zmq.Socket): The ZeroMQ subscriber socket.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        hwm: int = 1,
        conflate: bool = True,
        timeout: int = 1000,
    ) -> None:
        """
        Subscriber initialization.

        Args:
            host (str): The host address of the subscriber.
            port (int): The port number of the subscriber.
            hwm (int): High water mark for the subscriber. Default is 1.
            conflate (bool): Whether to conflate messages. Default is True.
            timeout (int): Maximum time to wait for a message in milliseconds. Default is 100 ms.
        """

        print("{:-^80}".format(" Phone Subscriber Initialization "))
        print(f"Address: tcp://{host}:{port}")

        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ subscriber
        self.subscriber = self.context.socket(zmq.SUB)
        # Set high water mark
        self.subscriber.set_hwm(hwm)
        # Set conflate
        self.subscriber.setsockopt(zmq.CONFLATE, conflate)
        # Connect the address
        self.subscriber.connect(f"tcp://{host}:{port}")
        # Subscribe the topic
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        # Set poller
        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)
        self.timeout = timeout

        # # Read the protobuf definition for Phone message
        # with open(
        #     pathlib.Path(__file__).parent / "phone_msg.proto",
        # ) as f:
        #     lines = f.read()
        # messages = re.search(r"message\s+Phone\s*{{(.*?)}}", lines, re.DOTALL)
        # body = messages.group(1)
        # print("Message Phone")
        # print("{\n" + body + "\n}")

        print("Phone Subscriber Initialization Done.")
        print("{:-^80}".format(""))

    def subscribeMessage(self) -> Tuple[float, bytes, bytes, int, int, list, list]:
        """
        Subscribe the message.

        Returns:
            timestamp (float): The timestamp of the message.
            color_img (bytes): The color image captured by the phone.
            depth_img (bytes): The depth image captured by the phone.
            depth_width (int): The width of the depth image.
            depth_height (int): The height of the depth image.
            local_pose (list): The local pose of the phone.
            global_pose (list): The global pose of the phone.

        Raises:
            zmq.ZMQError: If no message is received within the timeout period.
        """

        # Receive the message

        if self.poller.poll(self.timeout):
            # Receive the message
            msg = self.subscriber.recv()

            # Parse the message
            phone = phone_msg_pb2.Phone()
            phone.ParseFromString(msg)
        else:
            raise RuntimeError("No message received within the timeout period.")
        return (
            phone.timestamp,
            phone.color_img,
            phone.depth_img,
            phone.depth_width,
            phone.depth_height,
            phone.local_pose,
            phone.global_pose,
        )

    def close(self):
        """
        Close ZMQ socket and context.
        """

        if hasattr(self, "subscriber") and self.subscriber:
            self.subscriber.close()
        if hasattr(self, "context") and self.context:
            self.context.term()
